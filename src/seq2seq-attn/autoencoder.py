# -*- coding: utf-8 -*-
import os
import torch
import random
from utils import *
import numpy as np
import traceback
from io import open
from utils import *
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from config import model_config as conf

from tensorboardX import SummaryWriter

from attn import Attn
from encoder import EncoderRNN
from seq2seq import Sequence2SequenceNetwork


class FeatureAutoEncoderNetwork(Sequence2SequenceNetwork):
    # This autoencoder is to be used only for video and speech vectors
    # Use base Sequence2SequenceNetwork class for autoencoding text
    def build_model(self):
        # Note: no embedding used here
        self.encoder = EncoderRNN(self.enc_input_dim, self.hidden_size,
                                  self.enc_n_layers, self.dropout,
                                  self.unit, self.modality).to(self.device)

        self.encoder_optimizer = optim.Adam(self.encoder.parameters(),
                                            lr=self.lr)

        self.epoch = 0  # define here to add resume training feature

    def load_pretrained_model(self):
        if self.load_model_name:
            checkpoint = torch.load(self.load_model_name,
                                    map_location=self.device)
            print('Loaded {}'.format(self.load_model_name))
            self.epoch = checkpoint['epoch']
            self.encoder.load_state_dict(checkpoint['en'])
            self.encoder_optimizer.load_state_dict(checkpoint['en_op'])

    def train_model(self):
        best_score = 1e-200
        plot_losses = []
        print_loss_total = 0  # Reset every epoch

        start = time.time()
        saving_skipped = 0
        for epoch in range(self.epoch, self.n_epochs):
            random.shuffle(self.pairs)
            for iter in range(0, self.n_iters, self.batch_size):
                training_batch = batch2TrainData(
                    self.vocab, self.pairs[iter: iter + self.batch_size],
                    self.modality)

                if len(training_batch[1]) < self.batch_size:
                    print('skipped a batch..')
                    continue

                # Extract fields from batch
                input_variable, lengths, target_variable, \
                    tar_lengths = training_batch

                # Run a training iteration with the current batch
                loss = self.train(input_variable, lengths, target_variable, iter)
                self.writer.add_scalar('{}loss'.format(self.data_dir), loss, iter)

                print_loss_total += loss

            print_loss_avg = print_loss_total * self.batch_size / self.n_iters
            print_loss_total = 0
            print('Epoch: [{}/{}] Loss: {:.4f}'.format(
                epoch, self.n_epochs, print_loss_avg))

            if self.modality == 'tt':
                # evaluate and save the model
                curr_score = self.evaluate_all()
            else:  # ss, vv
                curr_score = print_loss_avg

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

    def train(self, input_variable, lengths, target_variable, iter):
        input_variable = input_variable.to(self.device)
        lengths = lengths.to(self.device)
        target_variable = target_variable.to(self.device)

        # Initialize variables
        loss = 0
        print_losses = []
        n_totals = 0

        # Forward pass through encoder
        encoder_outputs, encoder_hidden = self.encoder(input_variable, lengths)
        if self.unit == 'gru':
            latent = encoder_hidden
        else:
            (latent, cell_state) = encoder_hidden
        # reconstruct input from latent vector
        seq_len = input_variable.shape[0]
        self.latent2output = nn.Linear(self.latent_dim,
                                       self.enc_input_dim*seq_len).to(self.device)
        output = self.latent2output(latent)
        output = output.view(seq_len, self.batch_size, self.enc_input_dim)
        reconstructed_input = output

        loss = self.mean_square_error(reconstructed_input, target_variable)
        loss.backward()
        # Clip gradients: gradients are modified in place
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.clip)
        self.encoder_optimizer.step()
        return loss.item()

    def mean_square_error(self, inp, target):
        criterion = nn.MSELoss()
        inp = (inp.permute(1, 0, 2))
        target = (target.permute(1, 0, 2))
        return criterion(inp, target)

    def save_model(self, epoch):
        directory = self.save_dir
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save({
            'epoch': epoch,
            'en': self.encoder.state_dict(),
            'en_op': self.encoder_optimizer.state_dict()},
            '{}{}-{}-{}-{}.pth'.format(directory, self.model_code,
                                       self.modality, self.langs, epoch))


if __name__ == '__main__':
    FeatureAutoEncoderNetwork(conf)
