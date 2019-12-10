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
from decoder import DecoderRNN, GreedySearchDecoder


class Sequence2SequenceNetwork(object):
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

    def init_writer(self):
        self.writer = SummaryWriter()

    def load_configuration(self, config):
        # Load configuration
        self.iter_num = 0
        self.lr = config['lr']
        self.gpu = config['gpu']
        self.unit = config['unit']
        self.clip = config['clip']
        self.beta1 = config['beta1']
        self.beta2 = config['beta2']
        self.langs = config['langs']
        self.fusion = config['fusion']
        self.log_tb = config['log_tb']
        self.epsilon = config['epsilon']
        self.attn_model = config['attn']
        self.dropout = config['dropout']
        self.emb_mode = config['emb_mode']
        self.save_dir = config['save_dir']
        self.data_dir = config['data_dir']
        self.n_epochs = config['n_epochs']
        self.SOS_TOKEN = config['SOS_TOKEN']
        self.EOS_TOKEN = config['EOS_TOKEN']
        self.MAX_LENGTH = config['MAX_LENGTH']
        self.latent_dim = config['latent_dim']
        self.batch_size = config['batch_size']
        self.model_code = config['model_code']
        self.vocab_path = config['vocab_path']
        self.hidden_size = config['hidden_size']
        self.use_cuda = torch.cuda.is_available()
        self.log_tb_every = config['log_tb_every']
        self.enc_n_layers = config['enc_n_layers']
        self.dec_n_layers = config['dec_n_layers']
        self.dec_learning_ratio = config['dec_lr']
        self.bidirectional = config['bidirectional']
        self.enc_input_dim = config['enc_input_dim']
        self.embedding_dim = config['embedding_dim']
        self.use_scheduler = config['use_scheduler']
        self.use_embeddings = config['use_embeddings']
        self.lr_lower_bound = config['lr_lower_bound']
        self.teacher_forcing_ratio = config['tf_ratio']
        self.load_model_name = config['load_model_name']
        self.modality = config['modalities']  # no splitting as it's not multimodal case
        if self.modality in ['ss-vv', 'v-s']:
            self.pretrained_modality = config['pretrained_modality']
        self.generate_word_embeddings = config['generate_word_embeddings']
        self.device = torch.device(
            'cuda:{}'.format(self.gpu) if self.use_cuda else 'cpu')

    def load_vocabulary(self):
        try:
            with open(self.vocab_path, 'rb') as f:
                self.vocab = pickle.load(f)
        except FileNotFoundError as e:  # build vocab if it doesn't exist
            self.vocab = buildVocab()

    def prepare_data(self):
        # Note: The below workaround is used a lot and doing so is okay
        # because this script would only be run for unimodal cases
        self.pairs = prepareData(self.langs, [self.modality])[self.modality]
        num_pairs = len(self.pairs)
        self.pairs = self.pairs[: self.batch_size * (
            num_pairs // self.batch_size)]
        random.shuffle(self.pairs)
        self.n_iters = len(self.pairs)
        print('\nLoading test data pairs')
        self.test_pairs = prepareData(self.langs, [self.modality],
                                      train=False)[self.modality]
        random.shuffle(self.test_pairs)
        print(random.choice(self.pairs))
        if self.use_embeddings:
            if self.generate_word_embeddings:
                self.embedding_wts = generateWordEmbeddings(self.vocab,
                                                            self.emb_mode)
            else:
                self.embedding_wts = loadWordEmbeddings(self.emb_mode)

    def build_model(self):
        if self.use_embeddings:
            self.embedding = nn.Embedding.from_pretrained(self.embedding_wts)
        else:
            self.embedding = nn.Embedding(self.vocab.n_words,
                                          self.embedding_dim)

        if self.modality == 't':  # Need embedding only for t2t mode
            self.encoder = EncoderRNN(self.embedding_dim, self.hidden_size,
                                      self.enc_n_layers, self.dropout,
                                      self.unit, self.modality,
                                      self.embedding, fusion_or_unimodal=True).to(self.device)
        else:
            # Note: no embedding used here
            self.encoder = EncoderRNN(self.enc_input_dim, self.hidden_size,
                                      self.enc_n_layers, self.dropout,
                                      self.unit, self.modality, fusion_or_unimodal=True).to(self.device)

        self.decoder = DecoderRNN(self.attn_model, self.embedding_dim,
                                  self.hidden_size, self.vocab.n_words,
                                  self.unit, self.dec_n_layers,
                                  self.dropout, self.embedding).to(self.device)
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(),
                                            lr=self.lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(),
                                            lr=self.lr*self.dec_learning_ratio)

        self.epoch = 0  # define here to add resume training feature
        self.project_factor = self.encoder.project_factor
        self.latent2hidden = nn.Linear(
            self.latent_dim, self.hidden_size*self.project_factor).to(
            self.device)

    def load_pretrained_model(self):
        if self.load_model_name:
            checkpoint = torch.load(self.load_model_name,
                                    map_location=self.device)
            print('Loaded {}'.format(self.load_model_name))
            self.epoch = checkpoint['epoch']
            self.encoder.load_state_dict(checkpoint['en'])
            self.decoder.load_state_dict(checkpoint['de'])
            self.encoder_optimizer.load_state_dict(checkpoint['en_op'])
            self.decoder_optimizer.load_state_dict(checkpoint['de_op'])
            self.embedding.load_state_dict(checkpoint['embedding'])

    def train_model(self):
        best_score = 1e-200
        print_loss_total = 0  # Reset every epoch

        saving_skipped = 0
        for epoch in range(self.epoch, self.n_epochs):
            incomplete = False
            for iter in range(0, self.n_iters, self.batch_size):
                pairs = self.pairs[iter: iter + self.batch_size]
                # Skip incomplete batch
                if len(pairs) < self.batch_size:
                    incomplete = True
                    continue
                training_batch = batch2TrainData(
                    self.vocab, pairs, self.modality)

                # Extract fields from batch
                input_variable, lengths, target_variable, \
                    mask, max_target_len, _ = training_batch

                if incomplete:
                    break

                # Run a training iteration with the current batch
                loss = self.train(input_variable, lengths, target_variable,
                                  mask, max_target_len, iter)
                self.writer.add_scalar('{}loss'.format(self.data_dir), loss, iter)

                print_loss_total += loss

            print_loss_avg = print_loss_total * self.batch_size / self.n_iters
            print_loss_total = 0
            print('Epoch: [{}/{}] Loss: {:.4f}'.format(
                epoch, self.n_epochs, print_loss_avg))

            # evaluate and save the model
            curr_score = self.evaluate_all()

            self.writer.add_scalar('{}bleu_score'.format(self.data_dir), curr_score)

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

    def train(self, input_variable, lengths, target_variable, mask,
              max_target_len, iter):
        self.encoder.train()
        self.decoder.train()
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        input_variable = input_variable.to(self.device)
        lengths = lengths.to(self.device)
        target_variable = target_variable.to(self.device)
        mask = mask.to(self.device)

        # Initialize variables
        loss = 0
        print_losses = []
        n_totals = 0

        # Forward pass through encoder
        encoder_outputs, encoder_hidden = self.encoder(input_variable, lengths)

        # Create initial decoder input (start with SOS tokens for each sentence)
        decoder_input = torch.LongTensor([[self.SOS_TOKEN] * self.batch_size])
        decoder_input = decoder_input.to(self.device)

        # Set initial decoder hidden state to the encoder's final hidden state
        if self.unit == 'gru':
            decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        else:
            decoder_hidden = (encoder_hidden[0][:self.decoder.n_layers],
                              encoder_hidden[1][:self.decoder.n_layers])
        if iter % conf['log_tb_every'] == 0:
            # Visualize latent space
            if self.unit == 'gru':
                vis_hidden = decoder_hidden[-1, :, :]
            else:
                vis_hidden = decoder_hidden[0][-1, :, :]
            self.writer.add_embedding(vis_hidden,
                                      tag='decoder_hidden_{}'.format(iter))

        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

        if use_teacher_forcing:
            for t in range(max_target_len):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                # Teacher forcing: next input is current target
                decoder_input = target_variable[t].view(1, -1)
                # Calculate and accumulate loss
                mask_loss, nTotal = self.mask_nll_loss(decoder_output,
                                                       target_variable[t],
                                                       mask[t])
                loss += mask_loss
                print_losses.append(mask_loss.item() * nTotal)
                n_totals += nTotal
        else:
            for t in range(max_target_len):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                # No teacher forcing: next input is decoder's own current output
                _, topi = decoder_output.topk(1)
                decoder_input = torch.LongTensor(
                    [[topi[i][0] for i in range(self.batch_size)]])
                decoder_input = decoder_input.to(self.device)
                # Calculate and accumulate loss
                mask_loss, nTotal = self.mask_nll_loss(
                    decoder_output, target_variable[t], mask[t])
                loss += mask_loss
                print_losses.append(mask_loss.item() * nTotal)
                n_totals += nTotal

        loss.backward()

        # Clip gradients: gradients are modified in place
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.clip)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.clip)

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        return sum(print_losses) / n_totals

    def mask_nll_loss(self, inp, target, mask):
        n_total = mask.sum()
        cross_entropy = -torch.log(torch.gather(inp, 1,
                                   target.view(-1, 1)).squeeze(1))
        loss = cross_entropy.masked_select(mask).sum()
        loss = loss.to(self.device)
        return loss, n_total.item()

    def save_model(self, epoch):
        directory = self.save_dir
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save({
            'epoch': epoch,
            'en': self.encoder.state_dict(),
            'de': self.decoder.state_dict(),
            'en_op': self.encoder_optimizer.state_dict(),
            'de_op': self.decoder_optimizer.state_dict(),
            'embedding': self.embedding.state_dict()},
            '{}{}-{}-{}-{}.pth'.format(directory, self.model_code,
                                       self.modality, self.langs, epoch))

    def evaluate_all(self):
        self.encoder.eval()
        self.decoder.eval()
        searcher = GreedySearchDecoder(self.encoder, self.decoder,
                                       None, self.device,
                                       self.SOS_TOKEN)
        refs = []
        hyp = []
        for pair in self.test_pairs:
            output_words = self.evaluate(self.encoder, self.decoder,
                                         searcher, self.vocab, pair[0])
            if output_words:
                final_output = []
                for x in output_words:
                    if x == '<EOS>':
                        break
                    final_output.append(x)
                refs.append([pair[1].split()])
                hyp.append(final_output)
        bleu_scores = calculateBleuScores(refs, hyp)
        print('Bleu score: {bleu_1} | {bleu_2} | {bleu_3} | {bleu_4}'.format(
            **bleu_scores))
        eg_idx = random.choice(range(len(hyp)))
        print(hyp[eg_idx], refs[eg_idx])
        return bleu_scores['bleu_4']

    def evaluate(self, encoder, decoder, searcher, vocab, sentence_or_vector,
                 max_length=conf['MAX_LENGTH']):
        with torch.no_grad():
            if self.modality == 't':  # `sentence_or_vector` ~> sentence
                # Format input sentence as a batch
                # words => indexes
                indexes_batch = [indexesFromSentence(vocab, sentence_or_vector)]
                if None in indexes_batch:
                    return None
                for idx, indexes in enumerate(indexes_batch):
                    indexes_batch[idx] = indexes_batch[idx] + [self.EOS_TOKEN]
                # Create lengths tensor
                lengths = torch.tensor(
                    [len(indexes) for indexes in indexes_batch])
                # Transpose dimensions of batch to match models' expectations
                input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
            else:  # `sentence_or_vector` ~> vector
                input_batch, lengths = inputVarVec([sentence_or_vector], self.modality)

            # Use appropriate device
            input_batch = input_batch.to(self.device)
            lengths = lengths.to(self.device)
            # Decode sentence with searcher
            tokens, scores = searcher(input_batch, lengths, max_length)
            # indexes -> words
            decoded_words = [vocab.index2word[token.item()] for token in tokens]
            return decoded_words

    def close_writer(self):
        self.writer.close()


if __name__ == '__main__':
    Sequence2SequenceNetwork(conf)
