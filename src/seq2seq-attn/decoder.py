# -*- coding: utf-8 -*-
"""
This module contains different decoders
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import model_config as conf

from attn import Attn


class DecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding_dim, hidden_size, output_size,
                 unit='gru', n_layers=1, dropout=0.1, embedding=None,
                 latent_dim=300, bidirectional=True):
        super(DecoderRNN, self).__init__()

        self.unit = unit
        self.softmax = F.softmax
        self.n_layers = n_layers
        self.attn_model = attn_model
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.bidirectional = bidirectional
        self.embedding_dropout = nn.Dropout(dropout)
        self.dropout = (0 if n_layers == 1 else dropout)
        self.out = nn.Linear(self.hidden_size, self.output_size)

        if embedding:
            self.embedding = embedding
        if unit == 'gru':
            self.rnn = nn.GRU(embedding_dim, hidden_size,
                              n_layers, dropout=self.dropout)
        else:
            self.rnn = nn.LSTM(embedding_dim, hidden_size,
                               n_layers, dropout=self.dropout)

        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        if attn_model:
            self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        batch_size = input_step.shape[1]
        if self.embedding:
            # This is run one step at a time
            # Get current input words's embedding
            embedded = self.embedding(input_step)
            embedded = self.embedding_dropout(embedded)
            # Forward pass through unidirectional RNN
            rnn_output, hidden = self.rnn(embedded, last_hidden)
        else:
            rnn_output, hidden = self.rnn(input_step, last_hidden)

        if self.attn_model:
            # Get attention weights
            attn_weights = self.attn(rnn_output, encoder_outputs)
            # Get weighted sum
            context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
            # Concatenate weighted context vector and rnn output using Luong eq. 5
            rnn_output = rnn_output.squeeze(0)
            context = context.squeeze(1)
            concat_input = torch.cat((rnn_output, context), 1)
            concat_output = torch.tanh(self.concat(concat_input))
            # Predict next word (Luong eq. 6)
            output = self.out(concat_output)
        else:
            output = self.out(rnn_output[0])

        output = self.softmax(output, dim=1)
        # return output and final hidden state
        return output, hidden


class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder, latent2hidden, device, SOS_TOKEN):
        super(GreedySearchDecoder, self).__init__()
        self.device = device
        self.encoder = encoder
        self.decoder = decoder
        self.SOS_TOKEN = SOS_TOKEN
        self.latent2hidden = latent2hidden

    def forward(self, input_seq, input_length, max_length):
        batch_size = conf['batch_size']  # TODO: Make more elegant
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)

        # TODO: Refactor. encoder_hidden assignment at the end of the block is
        # duplication of the block above.
        if self.latent2hidden:
            if self.encoder.unit == 'gru':
                latent = encoder_hidden
            else:  # lstm
                (latent, cell_state) = encoder_hidden
                cell_state = cell_state[:self.decoder.n_layers]
            # Get back flattened hidden state
            hidden = self.latent2hidden(latent)
            # Unflatten hidden state
            if self.encoder.bidirectional or self.encoder.n_layers > 1:
                hidden = hidden.view(self.encoder.project_factor, batch_size,
                                     self.encoder.hidden_size)
            else:
                hidden = hidden.unsqueeze(0)
            hidden = hidden[:self.decoder.n_layers]
            encoder_hidden = hidden if self.encoder.unit == 'gru' else (hidden, cell_state)
            decoder_hidden = encoder_hidden
        else:  # for fusion experiments
            if self.encoder.unit == 'gru':
                decoder_hidden = encoder_hidden[:self.decoder.n_layers]
            elif self.encoder.unit == 'lstm':
                decoder_hidden = (encoder_hidden[0][:self.decoder.n_layers],
                                  encoder_hidden[1][:self.decoder.n_layers])
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(
            1, 1, device=self.device, dtype=torch.long) * self.SOS_TOKEN
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=self.device, dtype=torch.long)
        all_scores = torch.zeros([0], device=self.device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input,
                                                          decoder_hidden,
                                                          encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores
