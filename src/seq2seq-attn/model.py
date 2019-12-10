from seq2seq import *
from corrnet import *
from fusion_net import *
from autoencoder import *
from config import model_config as conf


def run_exp(config):
    if config['modalities'] in ['s', 'v', 't']:
        Sequence2SequenceNetwork(config)
    elif config['modalities'] in ['s-t', 't-v', 's-t-v']:
        FusionNetwork(config)
    elif config['modalities'] in ['ss-vv', 's-v', 'sv']:
        CorrelationNetwork(config)
    elif config['modalities'] in ['ss', 'vv']:
        FeatureAutoEncoderNetwork(config)


if __name__ == '__main__':
    run_exp(conf)
