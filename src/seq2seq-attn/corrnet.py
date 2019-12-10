# -*- coding: utf-8 -*-
import os
import torch
import random
import numpy as np
from utils import *
from io import open
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from config import model_config as conf

from tensorboardX import SummaryWriter

from attn import Attn
from encoder import EncoderRNN
from seq2seq import Sequence2SequenceNetwork
from decoder import DecoderRNN, GreedySearchDecoder


class CorrelationNetwork(Sequence2SequenceNetwork):
    def __init__(self, config):
        self.init_writer()
        self.load_configuration(config)
        self.load_vocabulary()
        self.prepare_data()
        self.build_model()
        self.load_pretrained_model()
        self.train_model()
        self.save_model(self.n_epochs)
        self.evaluate_all()
        self.close_writer()

    def build_model(self):
        if self.use_embeddings:
            self.embedding = nn.Embedding.from_pretrained(self.embedding_wts)
        else:
            self.embedding = nn.Embedding(self.vocab.n_words,
                                          self.embedding_dim)
        self.encoders = []
        self.encoder_optimizers = []

        # Note: No embeddings used in the encoders
        for m in ['v', 's']:
            encoder = EncoderRNN(self.enc_input_dim[m], self.hidden_size,
                                 self.enc_n_layers, self.dropout,
                                 self.unit, m).to(self.device)
            encoder_optimizer = optim.Adam(encoder.parameters(), lr=self.lr)

            if self.modality == 'ss-vv':
                checkpoint = torch.load(self.pretrained_modality[m],
                                        map_location=self.device)
                encoder.load_state_dict(checkpoint['en'])
                encoder_optimizer.load_state_dict(checkpoint['en_op'])
            self.encoders.append(encoder)
            self.encoder_optimizers.append(encoder_optimizer)
        self.decoder = DecoderRNN(self.attn_model, self.embedding_dim,
                                  self.hidden_size, self.vocab.n_words,
                                  self.unit, self.dec_n_layers,
                                  self.dropout, self.embedding).to(self.device)
        text_checkpoint = torch.load(self.pretrained_modality['t'],
                                     map_location=self.device)
        self.decoder.load_state_dict(text_checkpoint['de'])
        self.project_factor = self.encoders[0].project_factor
        self.latent2hidden = nn.Linear(
            self.latent_dim, self.hidden_size*self.project_factor).to(
            self.device)
        self.epoch = 0

    def train_model(self):
        best_score = 1e-200
        plot_losses = []
        print_loss_total = 0  # Reset every epoch

        saving_skipped = 0
        for epoch in range(self.epoch, self.n_epochs):
            random.shuffle(self.pairs)
            for iter in range(0, self.n_iters, self.batch_size):
                training_batch = batch2TrainData(
                    self.vocab, self.pairs[iter: iter + self.batch_size],
                    self.modality)
                # Extract fields from batch
                vid_vec, lengths, speech_vec, tar_lengths = training_batch

                # Run a training iteration with the current batch
                loss = self.train(vid_vec, lengths, speech_vec, tar_lengths, iter)
                self.writer.add_scalar('{}loss'.format(self.data_dir), loss, iter)

                print_loss_total += loss

            print_loss_avg = print_loss_total * self.batch_size / self.n_iters
            print_loss_total = 0
            print('Epoch: [{}/{}] Loss: {:.4f}'.format(
                epoch, self.n_epochs, print_loss_avg))

            # evaluate and save the model
            curr_score = self.evaluate_all()
            self.writer.add_scalar('{}bleu_score'.format(self.data_dir),
                                   curr_score)

            if curr_score > best_score:
                saving_skipped = 0
                best_score = curr_score
                self.save_model(epoch)

            saving_skipped += 1
            if self.use_scheduler and saving_skipped > 3:
                saving_skipped = 0
                new_lr = self.lr * 0.5
                print('Entered the dungeon...')
                if new_lr > self.lr_lower_bound:  # lower bound on lr
                    self.lr = new_lr
                    print('lr decreased to => {}'.format(self.lr))

    def train(self, input_variable, lengths, target_variable, tar_lengths, iter):
        for i, _ in enumerate(self.encoders):
            self.encoders[i].train()
            self.encoders[i].zero_grad()
        input_variable = input_variable.to(self.device)
        lengths = lengths.to(self.device)
        target_variable = target_variable.to(self.device)
        tar_lengths = tar_lengths.to(self.device)

        # Initialize variables
        loss = 0
        print_losses = []
        n_totals = 0

        # Forward pass through encoder
        enc_out_1, enc_hidden_1 = self.encoders[0](input_variable, lengths)
        enc_out_2, enc_hidden_2 = self.encoders[1](target_variable, tar_lengths)

        if self.unit == 'gru':
            latent_1 = enc_hidden_1
            latent_2 = enc_hidden_2
        else:  # lstm
            (latent_1, cs_1) = enc_hidden_1
            (latent_2, cs_2) = enc_hidden_2

        loss = self.mean_square_error(latent_1, latent_2)
        loss.backward()

        # Clip gradients: gradients are modified in place
        for i, _ in enumerate(self.encoders):
            torch.nn.utils.clip_grad_norm_(self.encoders[i].parameters(), self.clip)
            self.encoder_optimizers[i].step()
        return loss.item()

    def mean_square_error(self, inp, target):
        criterion = nn.MSELoss()
        return criterion(inp, target)

    def save_model(self, epoch):
        directory = '{}'.format(self.save_dir)
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save({
            'epoch': epoch,
            'en_1': self.encoders[0].state_dict(),
            'en_2': self.encoders[1].state_dict(),
            'en_op1': self.encoder_optimizers[0].state_dict(),
            'en_op2': self.encoder_optimizers[1].state_dict(),
            'de': self.decoder.state_dict()},
            '{}{}-{}-{}.pth'.format(directory, self.modality, self.langs, epoch))

    def evaluate_all(self):
        for i, _ in enumerate(self.encoders):
            self.encoders[i].eval()
        self.decoder.eval()
        searcher = GreedySearchDecoder(self.encoders[0], self.decoder,
                                       self.latent2hidden, self.device,
                                       self.SOS_TOKEN)
        refs = []
        hyp = []
        for pair in self.test_pairs:
            output_words = self.evaluate(searcher, self.vocab, pair[0])
            if output_words:
                final_output = []
                for x in output_words:
                    if x == '<EOS>':
                        break
                    final_output.append(x)
                refs.append([pair[2].split()])
                hyp.append(final_output)
        bleu_scores = calculateBleuScores(refs, hyp)
        print('Bleu score: {bleu_1} | {bleu_2} | {bleu_3} | {bleu_4}'.format(
            **bleu_scores))
        eg_idx = random.choice(range(len(hyp)))
        print(hyp[eg_idx], refs[eg_idx])
        return bleu_scores['bleu_4']

    def evaluate(self, searcher, vocab, sentence_or_vector,
                 max_length=conf['MAX_LENGTH']):
        with torch.no_grad():
            input_batch, lengths = inputVarVec([sentence_or_vector], self.modality)
            # Use appropriate device
            input_batch = input_batch.to(self.device)
            lengths = lengths.to(self.device)
            # Decode sentence with searcher
            tokens, scores = searcher(input_batch, lengths, max_length)
            # indexes -> words
            decoded_words = [vocab.index2word[token.item()] for token in tokens]
            return decoded_words


if __name__ == '__main__':
    CorrelationNetwork(conf)
