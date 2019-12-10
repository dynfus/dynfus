## MMDL

## Instructions

- Modify `config.py` for the current experiment
- Run with `python src/seq2seq-attn/model.py`

---

## Directory structure

```
├── README.md                                               <- Description of the project with instructions
├── REFERNCES.md                                            <- Brief summary of all the relevant works
├── requirements.txt                                        <- The requirements file for reproducing this project
├── setup.py                                                <- Make this project pip installable with `pip install -e`
├── src                                                     <- Source code for use in this project.
│   ├── seq2seq-attn                                        <- Directory for the sequence-to-sequence architechture
│   │   ├── config.py                                       <- Experiment settings
│   │   ├── kaldi_utils.py                                  <- Utils for processing kaldi audio vectors
│   │   ├── model.py                                        <- Seq2seq model with encoder, decoder and attn definitions
│   │   ├── plot.py                                         <- Visualization code
│   │   └── prepare_data.ipynb                              <- Notebook for preparing data for the experiments
│   ├── ARK_PATH                                            <- Directory for the audio vectors (copied from how2 dataset)
│   ├── data                                                <- General directory for storing data
│   │   ├── vocab.pkl                                       <- Vocab file for all the different modes
│   │   ├── s2t                                             <- Data files for s2t mode
│   │   |   ├── fbank_pitch_181506                          <- scp files for audio vectors
│   │   |   |   ├── train.cmvn.scp                          <- Copy cmvn.scp in 't2t/data/text-en(pt)/train/cmvn.scp' in how2 dataset as train.cmvn.scp
│   │   |   |   ├── val.cmvn.scp                            <- Similar as train.cmvn.scp
│   │   |   |   ├── dev5.cmvn.scp                           <- Similar as train.cmvn.scp
│   │   |   |   ├── train.feats.scp                         <- Copy cmvn.scp in 't2t/data/text-en(pt)/train/feats.scp' in how2 dataset as train.feats.scp
│   │   |   |   ├── val.feats.scp                           <- Similar as train.feats.scp
│   │   |   |   └── dev5.feats.scp                          <- Similar as train.feats.scp
│   │   |   ├── dev5.feat_map.pkl                           <- feature to vector-address map got by running kaldi_utils.py
│   │   |   ├── train.feat_map.pkl                          <- feature to vector-address map got by running kaldi_utils.py
│   │   |   └── val.feat_map.pkl                            <- feature to vector-address map got by running kaldi_utils.py
│   │   ├── t2t                                             <- Data files for t2t mode
│   │   |   ├── text-en/                                    <- Copied directly from how2 dataset (not actually used in t2t mode training)
│   │   |   ├── text-pt/                                    <- Copied directly from how2 dataset (not actually used in t2t mode training)
│   │   |   ├── en-pt.txt                                   <- Training file for t2t mode (got by running prepare_data.ipynb)
│   │   |   └── en-pt-test.txt                              <- Testing file for t2t mode (got by running prepare_data.ipynb)
│   │   ├── v2t                                             <- Data files for v2t mode
│   │   |   ├── resnext101-action-avgpool-300h              <- Copied directly from how2 dataset
│   │   |   |   ├── dev5.npy                                <- dev5 video vectors
│   │   |   |   ├── train.npy                               <- train video vectors
│   │   |   └── └── val.npy                                 <- val video vectors
│   ├── fasttext_emb                                        <- Word vectors for the experiment
│   │   ├── original                                        <- Official fasttext vectors
│   │   |   ├── wiki.en.vec                                 <- Fasttext vectors for English
│   │   |   └── wiki.pt.vec                                 <- Fasttext vectors for Portuguese
│   │   ├── s2t-en-w2v.pkl                                  <- Unfiltered en w2v embeddings for s2t mode
│   │   ├── s2t-pt-w2v.pkl                                  <- Unfiltered pt w2v embeddings for s2t mode
│   │   ├── t2t-en-w2v.pkl                                  <- Unfiltered en w2v embeddings for t2t mode
│   │   ├── t2t-pt-w2v.pkl                                  <- Unfiltered pt w2v embeddings for t2t mode
│   │   ├── v2t-en-w2v.pkl                                  <- Unfiltered en w2v embeddings for v2t mode
│   │   ├── v2t-pt-w2v.pkl                                  <- Unfiltered pt w2v embeddings for v2t mode
│   │   ├── s2t-en-pt-w2v-filtered_embeddings.pkl           <- Filtered w2v embeddings for s2t mode
│   │   ├── s2t-en-pt-fasttext-filtered_embeddings.pkl      <- Filtered fasttext embeddings for s2t mode
│   │   ├── v2t-en-pt-w2v-filtered_embeddings.pkl           <- Filtered w2v embeddings for v2t mode
│   │   ├── v2t-en-pt-fasttext-filtered_embeddings.pkl      <- Filtered fasttext embeddings for v2t mode
└── └── └── t2t-en-pt-w2v-filtered_embeddings.pkl           <- Filtered w2v embeddings for t2t mode
└── └── └── t2t-en-pt-fasttext-filtered_embeddings.pkl      <- Filtered fasttext embeddings for t2t mode
```

## Data Preparation:

**Note:** `root ==> seq2seq-attn` in the following instructions.

### t2t mode:
1. Run `prepare_data.ipynb` to generate `en-pt.txt` and `en-pt-test.txt` at appropriate locations

### s2t mode:
1. Run `cp data/t2t/text-en/data/train/cmvn.scp data/s2t/fbank_pitch_181506/train.cmvn.scp` from root
2. Run `cp data/t2t/text-en/data/train/feats.scp data/s2t/fbank_pitch_181506/train.feats.scp` from root
3. Repeat 1. & 2. for all modes (train/dev5/val) so that there are 6 `.scp` files generated inside `fbank_pitch_181506`
4. Run `python kaldi_utils.py` at root to generate the three `*.feats_scp.pkl` files

### v2t mode:
1. Nothing special. Just make sure the directory structure for `t2t` and `v2t` is followed

## Running the model:

1. When running for the very first time, make sure that `generate_embeddings` flag is `True` (Flip it `False` for any subsequent runs)
2. Make necessary changes in `config.py`. A couple of things to note while doing so:
    - Pay special attention to the `exp_mode`, `enc_input_dim` (follow comments for clarification)
    - As for the rest of the hyperparams, we'll need to figure out which one turns out to be the best
3. Run `python model.py`

