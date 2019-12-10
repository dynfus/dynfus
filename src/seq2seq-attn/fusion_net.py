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
from encoder import MultimodalEncoderRNN
from seq2seq import Sequence2SequenceNetwork
from decoder import DecoderRNN, GreedySearchDecoder


class FusionNetwork(Sequence2SequenceNetwork):
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

    def prepare_data(self):
        self.modality = self.modality.split('-')
        self.pairs = prepareData(self.langs, self.modality)  # dict: m => pairs
        num_pairs = len(random.choice(list(self.pairs.values())))
        rand_indices = random.sample(list(range(num_pairs)), num_pairs)
        self.n_iters = num_pairs

        for m in self.modality:
            self.pairs[m] = self.pairs[m][: self.batch_size * (
                num_pairs // self.batch_size)]
            # Shuffle all modalities the same way
            self.pairs[m] = [p for p, _ in sorted(zip(self.pairs[m], rand_indices))]
            print(random.choice(self.pairs[m]))

        print('\nLoading test data pairs')
        self.test_pairs = prepareData(self.langs,  self.modality, train=False)
        self.num_test_pairs = len(random.choice(list(self.test_pairs.values())))
        rand_indices = random.sample(list(range(self.num_test_pairs)),
                                     self.num_test_pairs)

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

        self.encoder = MultimodalEncoderRNN(self.fusion, self.hidden_size,
                                            self.enc_n_layers, self.dropout,
                                            self.unit, self.modality,
                                            self.embedding,
                                            self.device).to(self.device)

        if self.fusion == 'early' or self.fusion is None:
            parameter_list = self.encoder.parameters()
        else:
            parameter_list = []
            for m in self.encoder.modalities:
                parameter_list += list(self.encoder.encoder[m].parameters())

        # Need to expand hidden layer according to # modalities for early fusion
        self.decoder = DecoderRNN(self.attn_model, self.embedding_dim,
                                  self.hidden_size, self.vocab.n_words,
                                  self.unit, self.dec_n_layers, self.dropout,
                                  self.embedding).to(self.device)
        self.encoder_optimizer = optim.Adam(parameter_list,
                                            lr=self.lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(),
                                            lr=self.lr*self.dec_learning_ratio)
        self.epoch = 0  # define here to add resume training feature

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
            self.vocab.__dict__ = checkpoint['vocab_dict']
            self.evaluate_all()

    def train_model(self):
        best_score = 1e-200
        print_loss_total = 0  # Reset every epoch

        num_pairs = {}
        for m in self.modality:
            num_pairs[m] = len(self.pairs[m])

        saving_skipped = 0
        for epoch in range(self.epoch, self.n_epochs):
            incomplete = False
            for iter in range(0, self.n_iters, self.batch_size):
                training_batch = {}
                input_variable = {}
                lengths = {}
                for m in self.modality:
                    pairs = self.pairs[m][iter: iter + self.batch_size]
                    # Skip incomplete batch
                    if len(pairs) < self.batch_size:
                        incomplete = True
                        continue
                    training_batch[m] = batch2TrainData(
                        self.vocab, pairs, m)
                    # Extract fields from batch
                    input_variable[m], lengths[m], target_variable, \
                        mask, max_target_len, _ = training_batch[m]

                if incomplete:
                    break

                # Run a training iteration with the current batch
                loss = self.train(input_variable, lengths, target_variable,
                                  mask, max_target_len, epoch, iter)
                self.writer.add_scalar('{}loss'.format(self.data_dir), loss, iter)

                print_loss_total += loss

            print_loss_avg = print_loss_total * self.batch_size / self.n_iters
            print_loss_total = 0
            print('Epoch: [{}/{}] Loss: {:.4f}'.format(
                epoch, self.n_epochs, print_loss_avg))

            # evaluate and save the model
            curr_score = self.evaluate_all()

            self.writer.add_scalar('{}bleu_score'.format(self.data_dir), curr_score, iter)
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
              max_target_len, epoch, iter):
        self.encoder.train()
        self.decoder.train()
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        for m in self.modality:
            input_variable[m] = input_variable[m].to(self.device)
            lengths[m] = lengths[m].to(self.device)
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
        if iter % conf['log_tb_every'] == 1:
            # Visualize latent space
            if self.unit == 'gru':
                vis_hidden = decoder_hidden[-1, :, :]
            else:
                vis_hidden = decoder_hidden[0][-1, :, :]
            self.writer.add_embedding(vis_hidden,
                                      tag='decoder_hidden_{}_{}'.format(
                                        epoch, iter))

        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

        if use_teacher_forcing:
            for t in range(max_target_len):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
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
            'vocab_dict': self.vocab.__dict__,
            'embedding': self.embedding.state_dict()},
            '{}{}-{}-{}.pth'.format(directory, self.model_code, epoch, iter))

    def evaluate_all(self):
        self.encoder.eval()
        self.decoder.eval()
        searcher = GreedySearchDecoder(
            self.encoder, self.decoder, None, self.device, self.SOS_TOKEN)
        refs = []
        hyp = []

        for id in range(self.num_test_pairs):
            # Sample test pairs of each modality
            output_words, reference = self.evaluate(
                searcher, self.vocab, self.test_pairs, id)
            if output_words:
                final_output = []
                for x in output_words:
                    if x == '<EOS>':
                        break
                    final_output.append(x)
                refs.append(reference.split())
                hyp.append(final_output)

        bleu_scores = calculateBleuScores(refs, hyp)
        print('Bleu score: {bleu_1} | {bleu_2} | {bleu_3} | {bleu_4}'.format(
            **bleu_scores))
        eg_idx = random.choice(range(len(hyp)))
        print(hyp[eg_idx], refs[eg_idx])
        return bleu_scores['bleu_4']

    def evaluate(self, searcher, vocab, test_pairs, id,
                 max_length=conf['MAX_LENGTH']):
        lengths = {}
        input_batch = {}
        with torch.no_grad():
            reference = random.choice(list(test_pairs.values()))[id][1]
            for m in self.modality:
                sentence_or_vector = test_pairs[m][id][0]
                if m == 't':  # `sentence_or_vector` ~> sentence
                    # Format input sentence as a batch
                    # words => indexes
                    indexes_batch = [indexesFromSentence(vocab, sentence_or_vector)]
                    if None in indexes_batch:
                        return None
                    for idx, indexes in enumerate(indexes_batch):
                        indexes_batch[idx] = indexes_batch[idx] + [self.EOS_TOKEN]
                    # Create lengths tensor
                    lengths[m] = torch.tensor(
                        [len(indexes) for indexes in indexes_batch])
                    # Transpose dimensions of batch to match models' expectations
                    input_batch[m] = torch.LongTensor(
                        indexes_batch).transpose(0, 1)
                else:  # `sentence_or_vector` ~> vector
                    input_batch[m], lengths[m] = \
                        inputVarVec([sentence_or_vector], m)

                # Use appropriate device
                input_batch[m] = input_batch[m].to(self.device)
                lengths[m] = lengths[m].to(self.device)

            # Decode sentence with searcher
            tokens, scores = searcher(input_batch, lengths, max_length)
            # indexes -> words
            decoded_words = [vocab.index2word[token.item()] for token in tokens]
            return decoded_words, reference

    def close_writer(self):
        self.writer.close()


if __name__ == '__main__':
    FusionNetwork(conf)
