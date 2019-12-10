import re
import time
import math
from io import open
import random
import nltk
import torch
import pickle
import random
import gensim
import itertools
import numpy as np
import unicodedata
import kaldi_io as ki
import kaldi_utils as ku
from config import model_config as conf
from torch.nn.utils.rnn import pad_sequence


class Vocabulary(object):
    def __init__(self, languages):
        self.word2index = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.word2count = {}
        self.index2word = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.n_words = 4  # Count PAD, SOS, EOS and UNK
        self.languages = languages
        self.index2lang = []

    def add_sentence(self, sentence, lang):
        for word in sentence.split():
            self.add_word(word, lang)

    def add_word(self, word, lang):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
            self.index2lang.append(lang)
        else:
            self.word2count[word] += 1


def buildVocab():
    langs = conf['langs']
    modality = conf['modalities']
    print('Loading corpus...')
    pairs = list(readPairs(langs).values())[0] + \
        list(readPairs(langs, train=False).values())[0]
    pairs = filterPairs(pairs)
    vocab = Vocabulary(langs)
    print("Building vocab...")
    l_split = langs.split('-')
    vocab = Vocabulary(langs)
    src_lang, target_lang = l_split[0], l_split[1]
    for pair in pairs:
        vocab.add_sentence(pair[0], src_lang)
        vocab.add_sentence(pair[1], target_lang)
    print("Total words in {} vocab:".format(vocab.languages))
    print(vocab.n_words)
    with open(conf['vocab_path'], 'wb') as f:
        pickle.dump(vocab, f)
    return vocab


def trainWord2VecModel():
    langs = conf['langs']
    src_lang, target_lang = langs.split('-')
    modality = conf['modalities']
    print('Loading corpus...')
    pairs = list(readPairs(langs).values())[0] + \
        list(readPairs(langs, train=False).values())[0]
    pairs = filterPairs(pairs)
    random.shuffle(pairs)

    sentences_l1 = []
    sentences_l2 = []
    for pair in pairs:
        sentences_l1.append(pair[0].split())
        sentences_l2.append(pair[1].split())

    print('Training word2vec for {}...'.format(src_lang))
    w2v = gensim.models.Word2Vec(sentences_l1, size=300, min_count=1, iter=50)
    with open('{}{}-w2v.pkl'.format(
              conf['embeddings_path'], src_lang), 'wb') as f:
        pickle.dump(w2v, f)

    print('Training word2vec for {}...'.format(target_lang))
    w2v = gensim.models.Word2Vec(sentences_l2, size=300, min_count=1, iter=50)
    with open('{}{}-w2v.pkl'.format(
              conf['embeddings_path'], target_lang), 'wb') as f:
        pickle.dump(w2v, f)
    print('w2v models trained.')


def generateWordEmbeddings(vocab, mode='w2v'):
    modality = conf['modalities']
    langs = vocab.languages
    modalities = conf['modalities']
    if 't' not in modalities:  # will have just one language
        langs = langs.split('-')[-1]
        print(langs, 'inside embeddings..')
        if mode == 'fasttext':
            embeddings_all = gensim.models.KeyedVectors.load_word2vec_format(
                '{}original/wiki.{}.vec'.format(conf['embeddings_path'], langs))
        elif mode == 'w2v':
            embeddings_all = gensim.models.Word2Vec.load(
                '{}{}-w2v.pkl'.format(conf['embeddings_path'], langs))

        print('Loaded original embeddings')

        # initialize word embeddings matrix
        combined_word_embeddings = np.zeros((vocab.n_words,
                                             conf['embedding_dim']))
        for index, word in vocab.index2word.items():
            try:
                if index < 4:  # deal with special tokens
                    combined_word_embeddings[index] = np.random.normal(
                        size=(conf['embedding_dim'], ))
                    continue
                combined_word_embeddings[index] = embeddings_all[word]
            except KeyError as e:
                print('KeyError triggered for {}'.format(word))
                combined_word_embeddings[index] = np.random.normal(
                    size=(conf['embedding_dim'], ))
    else:  # t2t mode
        src_lang, target_lang = langs.split('-')
        print(src_lang, target_lang, 'inside embeddings...')

        # Load appropriate embeddings
        if mode == 'fasttext':
            src_embeddings_all = gensim.models.KeyedVectors.load_word2vec_format(
                '{}original/wiki.{}.vec'.format(conf['embeddings_path'],
                                                src_lang))
            # Avoid loading the same (massive) embeddings file
            target_embeddings_all = \
                (src_embeddings_all if src_lang == target_lang else
                    gensim.models.KeyedVectors.load_word2vec_format(
                        '{}original/wiki.{}.vec'.format(
                            conf['embeddings_path'], target_lang)))
        elif mode == 'w2v':
            src_embeddings_all = \
                gensim.models.Word2Vec.load(
                    '{}{}-w2v.pkl'.format(conf['embeddings_path'], src_lang))

            target_embeddings_all = \
                (src_embeddings_all if src_lang == target_lang else
                    gensim.models.Word2Vec.load(
                       '{}{}-w2v.pkl'.format(conf['embeddings_path'], target_lang)))
        print('Loaded original embeddings.')

        combined_word_embeddings = np.zeros((vocab.n_words,
                                             conf['embedding_dim']))
        for index, word in vocab.index2word.items():
            try:
                if index < 4:  # deal with special tokens
                    combined_word_embeddings[index] = np.random.normal(
                        size=(conf['embedding_dim'], ))
                    continue
                # offset between index2lang and index2word is 4 due to spl tkns
                combined_word_embeddings[index] = src_embeddings_all[word] \
                    if vocab.index2lang[index - 4] == src_lang \
                    else target_embeddings_all[word]
            except KeyError as e:
                print('KeyError triggered for {}'.format(word))
                combined_word_embeddings[index] = np.random.normal(
                    size=(conf['embedding_dim'], ))

    print('Created combined + filtered embeddings.')
    with open('{}{}-{}-filtered_embeddings.pkl'.format(
            conf['embeddings_path'], langs, mode), 'wb') as f:
        pickle.dump(combined_word_embeddings, f)
    combined_word_embeddings = torch.from_numpy(combined_word_embeddings).float()
    return combined_word_embeddings


def loadWordEmbeddings(mode='w2v'):
    with open('{}{}-{}-filtered_embeddings.pkl'.format(
            conf['embeddings_path'], conf['langs'], mode), 'rb') as f:
        combined_word_embeddings = pickle.load(f)
        return torch.from_numpy(combined_word_embeddings).float()


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def zeroPadding(l, fillvalue=conf['PAD_TOKEN']):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))


def binaryMatrix(l, value=conf['PAD_TOKEN']):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == conf['PAD_TOKEN']:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


def readPairs(langs, train=True, modalities=['t']):
    """
    langs is without extension
    filtered=True to read preprocessed multimodal data
    """
    # Load source file
    pairs = {}
    for m in modalities:
        pairs[m] = []
        if m != 't':
            file_path = '{0}/t2t/text-{1}/{0}/{2}/text.id.{1}'.format(
                conf['data_dir'], langs.split('-')[-1],
                ('train' if train else 'val'))
        elif m == 't':
            file_path = '{}t2t/{}{}.txt'.format(
                conf['data_dir'], langs, ('-train' if train else '-val'))
        # Read the file and split into lines
        lines = open(file_path, encoding='utf-8').readlines()

        sep = '\t' if m == 't' else ' '
        for l in lines:
            s1, s2 = l.split(sep, 1)
            if s1 in ['D0T7ho08Q3o_25', 'FUfPuPMxh2w_9']:
                continue
            if m in ['s', 'v', 'vv', 'ss', 'ss-vv']:
                # s1 ~> ark file keys for `get_ark_rep`
                # s2 ~> target sentences, used in `ss-vv`, discarded in other
                # modes
                pairs[m].append([s1, normalizeString(s2)])
            else:  # 't'
                pairs[m].append([normalizeString(s1), normalizeString(s2)])

    return pairs


def shufflePairs(pairs, modalities):
    # Shuffle multimodal pair dict uniformly
    num_pairs = len(random.choice(list(pairs.values())))
    rand_indices = random.sample(list(range(num_pairs)), num_pairs)
    for m in modalities:
        pairs[m] = [p for p, _ in sorted(zip(pairs[m], rand_indices))]
    return pairs


def filterPairs(pairs):
    """Filter only based on target for multimodal"""
    return [pair for pair in pairs if len(pair[1].split()) < conf['MAX_LENGTH']]


def prepareData(langs, modalities=['t'], train=True):
    pairs = readPairs(langs, train, modalities)  # dict: modality -> pairs

    for m in modalities:
        print("Read {} sentence pairs for modality {}".format(len(pairs[m]), m))
        pairs[m] = filterPairs(pairs[m])
        # Only keep pairs that exist in all modalities
        print("Trimmed to {} sentence pairs for modality {}".format(len(pairs[m]), m))

    # Shuffle pairs the same way
    pairs = shufflePairs(pairs, modalities)

    if not train:  # fetch vectors directly for test mode
        for m in modalities:
            if m == 't':
                continue
            pairs[m] = getVectorizedPairBatch(pairs[m], m, train=False)
            if m in ['s', 'v']:
                pairs[m] = [[torch.FloatTensor(x), y] for x, y in pairs[m]]
            elif m in ['vv', 'ss', 'ss-vv']:
                if not train and m == 'ss-vv':
                    pairs[m] = [[torch.FloatTensor(x), torch.FloatTensor(y), z]
                                for x, y, z in pairs[m]]
                else:
                    pairs[m] = [[torch.FloatTensor(x), torch.FloatTensor(y)]
                                for x, y in pairs[m]]

    return pairs


def indexesFromSentence(vocab, sentence):
    indexes = []
    for word in sentence.split():
        try:
            indexes.append(vocab.word2index[word])
        except KeyError as e:  # OOV
            indexes.append(conf['UNK_TOKEN'])
    return indexes


def calculateBleuScores(references, hypothesis):
    weights = [0., 0., 0., 0.]
    bleu_scores = {}
    for i in range(1, 5):
        w = 1.0 / i
        for idx in range(i):
            weights[idx] = w
        bleu_scores['bleu_{}'.format(str(i))] = 100 * nltk.translate.bleu_score\
            .corpus_bleu(references, hypothesis, weights=weights)
    return bleu_scores


# Returns padded input sequence tensor and lengths
def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    for idx, indexes in enumerate(indexes_batch):
        indexes_batch[idx] = indexes_batch[idx] + [conf['EOS_TOKEN']]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths


def inputVarVec(indexes_batch, modality):
    # inputVar for n-dimensional vectors (for s2t and v2t)
    for idx, indexes in enumerate(indexes_batch):
        indexes_batch[idx] = torch.FloatTensor(indexes_batch[idx])

    # find lengths of sequences
    if modality == 'v':
        lengths = torch.ones(len(indexes_batch), dtype=torch.int32)
    else:
        lengths = torch.tensor([int(indexes.shape[0]) for indexes in indexes_batch])
    padVar = pad_sequence(indexes_batch, padding_value=conf['PAD_TOKEN'])
    return padVar, lengths


# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    for idx, indexes in enumerate(indexes_batch):
        indexes_batch[idx] = indexes_batch[idx] + [conf['EOS_TOKEN']]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len, lengths


def getVectorizedPairBatch(pair_batch, modality, train=True):
    # TODO: Refactor
    ftype = 'train' if train else 'val'
    if modality == 's':
        with open('{}s2t/{}.feat_map.pkl'.format(
                conf['data_dir'], ftype), 'rb') as f:
            feat_map = pickle.load(f)
        final_pair_batch = []
        for x, y in pair_batch:
            final_pair_batch.append(
                [ki.read_mat(ku.get_ark_rep(x, feat_map)), y])
        return final_pair_batch
    elif modality == 'ss':
        with open('{}s2t/{}.feat_map.pkl'.format(
                conf['data_dir'], ftype), 'rb') as f:
            feat_map = pickle.load(f)
        final_pair_batch = []
        for x, y in pair_batch:
            e = ki.read_mat(ku.get_ark_rep(x, feat_map))
            final_pair_batch.append([e, e])
        return final_pair_batch
    elif modality == 'v':
        video_feats = np.load(
            '{}v2t/resnext101-action-avgpool-300h/{}.npy'.format(
                conf['data_dir'], ftype))
        return [[video_feats[idx], y] for idx, (_, y) in enumerate(pair_batch)]
    elif modality == 'vv':
        video_feats = np.load(
            '{}v2t/resnext101-action-avgpool-300h/{}.npy'.format(
                conf['data_dir'], ftype))
        final_pair_batch = []
        for idx, (_, y) in enumerate(pair_batch):
            e = video_feats[idx]
            final_pair_batch.append([e, e])
        return final_pair_batch
    elif modality in ['sv', 'ss-vv']:
        # Load speech vectors
        with open('{}s2t/{}.feat_map.pkl'.format(
                conf['data_dir'], ftype), 'rb') as f:
            feat_map = pickle.load(f)
        # Load video vectors
        video_feats = np.load(
            '{}v2t/resnext101-action-avgpool-300h/{}.npy'.format(
                conf['data_dir'], ftype))
        final_pair_batch = []
        # Note that the text (t) is simply discarded for training mode
        for idx, (x, t) in enumerate(pair_batch):
            if train:
                final_pair_batch.append(
                    [video_feats[idx], ki.read_mat(ku.get_ark_rep(x, feat_map))])
            else:
                final_pair_batch.append(
                    [video_feats[idx],
                     ki.read_mat(ku.get_ark_rep(x, feat_map)),
                     t])
        return final_pair_batch


# Returns all items for a given batch of pairs
def batch2TrainData(voc, pair_batch, modality):
    input_batch, output_batch = [], []
    if modality in ['s', 'v']:
        pair_batch = getVectorizedPairBatch(pair_batch, modality)
        pair_batch.sort(key=lambda x: x[0].shape[0], reverse=True)
        for pair in pair_batch:
            input_batch.append(pair[0])
            output_batch.append(pair[1])
        inp, lengths = inputVarVec(input_batch, modality)
        output, mask, max_target_len, tar_lengths = outputVar(output_batch, voc)
        return inp, lengths, output, mask, max_target_len, tar_lengths
    elif modality in ['ss', 'vv']:
        pair_batch = getVectorizedPairBatch(pair_batch, modality)
        pair_batch.sort(key=lambda x: x[0].shape[0], reverse=True)
        for pair in pair_batch:
            input_batch.append(pair[0])
        inp, lengths = inputVarVec(input_batch, modality)
        return inp, lengths, inp, lengths
    elif modality == 'ss-vv':
        pair_batch = getVectorizedPairBatch(pair_batch, modality)  # [(vid_vec, speech_vec)]
        pair_batch.sort(key=lambda x: x[1].shape[0], reverse=True)
        for pair in pair_batch:
            input_batch.append(pair[0])
            output_batch.append(pair[1])
        inp, lengths = inputVarVec(input_batch, modality)
        out, tar_lengths = inputVarVec(output_batch, modality)
        return inp, lengths, out, tar_lengths

    pair_batch.sort(key=lambda x: len(x[0].split()), reverse=True)
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len, tar_lengths = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len, tar_lengths


if __name__ == '__main__':
    buildVocab()
    trainWord2VecModel()
