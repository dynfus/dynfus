# -*- coding: utf-8 -*-
"""
This module contains different types encoders
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import model_config as conf


class EncoderRNN(nn.Module):
    def __init__(self, input_dim, hidden_size, n_layers=1, dropout=0.1,
                 unit='gru', modality='t', embedding=None, latent_dim=300,
                 bidirectional=True, fusion_or_unimodal=False):
        super(EncoderRNN, self).__init__()
        self.unit = unit
        self.n_layers = n_layers
        self.modality = modality
        self.latent_dim = latent_dim
        self.fusion_or_unimodal = fusion_or_unimodal
        if modality == 't':
            self.embedding = embedding
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.bidirectional = bidirectional
        self.dropout = (0 if n_layers == 1 else dropout)
        self.project_factor = n_layers * (2 if bidirectional else 1)
        self.hidden2latent = nn.Linear(hidden_size * self.project_factor,
                                       latent_dim)
        # choose appropriate RNN
        if unit == 'gru':
            self.rnn = nn.GRU(input_dim, hidden_size, n_layers,
                              dropout=self.dropout, bidirectional=True)
        else:
            self.rnn = nn.LSTM(input_dim, hidden_size, n_layers,
                               dropout=self.dropout, bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        batch_size = input_seq.shape[1]
        if self.modality == 'v':  # video vectors
            input_seq = input_seq.unsqueeze(0).permute(0, 2, 1)
        if self.modality == 't':
            # Convert word indexes to embeddings
            embedded = self.embedding(input_seq)
            # Pack padded batch of sequences for RNN module
            packed = torch.nn.utils.rnn.pack_padded_sequence(embedded,
                                                             input_lengths)
        else:
            packed = torch.nn.utils.rnn.pack_padded_sequence(input_seq,
                                                             input_lengths)

        # Forward pass through RNN
        if self.fusion_or_unimodal:
            outputs, hidden = self.rnn(packed, hidden)
            outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
            outputs = outputs[:, :, :self.hidden_size] + \
                outputs[:, :, self.hidden_size:]
            return outputs, hidden
        else:  # non fusion experiments
            if self.unit == 'gru':
                outputs, hidden = self.rnn(packed, hidden)
            elif self.unit == 'lstm':  # lstm
                outputs, (hidden, cell_state) = self.rnn(packed, hidden)

        # flatten hidden state for projecting to the latent space
        if self.bidirectional or self.n_layers > 1:
            hidden = hidden.view(batch_size,
                                 self.hidden_size * self.project_factor)
        else:
            hidden = hidden.squeeze()
        latent = self.hidden2latent(hidden)
        last_hidden = latent if self.unit == 'gru' else (latent, cell_state)
        return outputs, last_hidden


class MultimodalEncoderRNN(nn.Module):
    def __init__(self, fusion_type, hidden_size, n_layers=1, dropout=0.1,
                 unit='gru', modalities=['t'], embedding=None,
                 device='cuda:0'):
        super(MultimodalEncoderRNN, self).__init__()
        self.fusion_type = fusion_type
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = (0 if n_layers == 1 else dropout)
        self.unit = unit
        self.device = device
        self.modalities = modalities
        if 't' in modalities:
            self.embedding = embedding
        else:
            self.embedding = None
        if self.fusion_type == 'early':
            # Concatenate all features
            input_dim = conf['enc_input_dim']
            self.encoder = EncoderRNN(
                input_dim, self.hidden_size, self.n_layers, self.dropout,
                self.unit, None, self.embedding, fusion_or_unimodal=True).to(self.device)
        # TODO: load pretrained unimodal subnetworks
        else:  # late or tfn
            self.encoder = {}
            for m in modalities:
                if m == 't':  # Need embedding only for t2t mode
                    self.encoder[m] = EncoderRNN(
                        conf['embedding_dim'], self.hidden_size, self.n_layers,
                        self.dropout, self.unit, m, self.embedding,
                        fusion_or_unimodal=True).to(self.device)
                else:  # Note: no embedding used here
                    self.encoder[m] = EncoderRNN(
                        conf['enc_input_dim'][m], self.hidden_size,
                        self.n_layers, self.dropout, self.unit, m,
                        fusion_or_unimodal=True).to(self.device)
            if self.fusion_type == 'late':
                self.post_fusion = nn.Linear(
                    len(self.modalities) * self.hidden_size, self.hidden_size)
            else:  # tfn
                exp_hidden_size = 1
                for _ in range(len(self.modalities)):
                    exp_hidden_size *= (self.hidden_size + 1)
                self.post_fusion = nn.Linear(exp_hidden_size, self.hidden_size)

    def forward(self, input_seq, input_lengths, hidden=None):
        if self.fusion_type == 'early':
            concat_input_seq = []
            concat_input_lengths = []
            # TODO: set max vector length dynamically (currently hardcoded)
            for m in self.modalities:
                if m == 't':
                    # Convert word indexes to embeddings
                    input_seq[m] = self.embedding(input_seq[m])
                elif m == 'v':
                    input_seq[m] = input_seq[m].unsqueeze(0).permute(0, 2, 1)
                # Swap axes
                input_seq[m] = input_seq[m].permute(2, 1, 0)
                # Pad sequence length
                padding = nn.ConstantPad1d((0, 1000 - input_seq[m].shape[2]), 0)
                input_seq[m] = padding(input_seq[m])
                # Swap axes back
                input_seq[m] = input_seq[m].permute(2, 1, 0)
                concat_input_seq.append(input_seq[m])
                concat_input_lengths.append(input_lengths[m])
            # Concatenate multimodal features
            concat_input_seq = torch.cat(concat_input_seq, dim=2)
            # Set max len for each batch
            concat_input_lengths = torch.LongTensor(
                [1000] * concat_input_seq.shape[1]).to(self.device)
            output, hidden = self.encoder(
                concat_input_seq, concat_input_lengths)
            return output, hidden
        elif self.fusion_type is None:
            if len(self.modalities) == 1:
                modality = self.modalities[0]
                output, hidden = self.encoder(
                    input_seq[modality], input_lengths[modality])
            else:
                raise ValueError('Incorrect fusion method chosen.')
        else:
            concat_hidden = []
            concat_output = []
            for m in self.modalities:
                output, hidden = self.encoder[m](input_seq[m], input_lengths[m])
                hidden_state = hidden[0] if self.unit == 'lstm' else hidden
                if self.fusion_type == 'tfn':
                    # Expand modality tensors by ones dimension
                    batch_size = hidden_state.shape[1]
                    hidden_state = torch.cat(
                        (torch.ones((2 * self.n_layers, batch_size, 1),
                         requires_grad=False, device=self.device), hidden_state),
                        dim=2)
                    # Convert into batch first tensor
                    hidden_state = hidden_state.permute(1, 0, 2)
                concat_hidden.append(hidden_state)
                concat_output.append(output)
            if self.fusion_type == 'late':
                # TODO: try weighted sum
                # Concatenate hidden states
                hidden_state = torch.cat(concat_hidden, dim=2)
                # Reshape to post fusion hidden size
                hidden_state = self.post_fusion(hidden_state)
            elif self.fusion_type == 'tfn':
                num_modalities = len(self.modalities)
                if num_modalities < 3:
                    raise Exception('Not enough modalities for TFN.')
                else:
                    # Take outer product of the first two modalities
                    fusion_tensor = torch.matmul(concat_hidden[0].unsqueeze(3),
                                                 concat_hidden[1].unsqueeze(2))
                    # Take the Knoecker product
                    fusion_tensor = fusion_tensor.view(-1, 2 * self.n_layers,
                            (self.hidden_size + 1) * (self.hidden_size + 1), 1)
                    fusion_tensor = torch.matmul(
                        fusion_tensor, concat_hidden[2].unsqueeze(2))
                    fusion_tensor = fusion_tensor.view(
                        fusion_tensor.shape[0], 2 * self.n_layers, -1)
                    # Return batch size to dim 1
                    fusion_tensor = fusion_tensor.permute(1, 0, 2)
                    # Reshape to post fusion hidden size
                    hidden_state = self.post_fusion(fusion_tensor)
            hidden = (hidden_state, hidden[1]) if self.unit == 'lstm' else hidden_state
            output = torch.cat(concat_output, dim=0)
        return output, hidden
