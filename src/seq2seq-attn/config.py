model_config = {
    'gpu': 0,  # id of the gpu to use
    'clip': 50.0,  # for clipping gradients
    'attn': None,  # attention mechanism to choose (dot/concat/general/none)
    'lr': 0.00001,  # learning rate
    'beta1': 0.9,  # Adam parameters
    'beta2': 0.999,
    'epsilon': 1e-8,
    'dec_lr': 5.0,  # decoder learning ratio
    'unit': 'lstm',  # gru or lstm
    'PAD_TOKEN': 0,
    'SOS_TOKEN': 1,
    'EOS_TOKEN': 2,
    'UNK_TOKEN': 3,
    'dropout': 0.3,
    'langs': 'en-pt',  # languages whose text would be involved in the exp
    'tf_ratio': 1.0,  # teacher forcing ratio
    'log_tb': False,  # turn on only for visualization runs
    'n_epochs': 1000,
    'batch_size': 64,
    'emb_mode': 'w2v',  # w2v/fasttext
    'enc_n_layers': 2,
    'dec_n_layers': 2,
    'latent_dim': 300,  # latent space dimension (for other experiment modes)
    'MAX_LENGTH': 100,
    'data_dir': 'data/',
    'hidden_size': 256,  # TODO: set different hidden sizes for each modality
    'log_tb_every': 1000,  # how often should we save embeddings to tensorboardx
    'modalities': 't',  # t/s/ss/v/vv/s-t/t-v/s-t-v/ss-vv/s-v/sv
    'fusion': None,  # None (unimodal), early, late or tfn
    'embedding_dim': 300,
    'use_scheduler': True,  # do lr/2 if perf. doesn't improve for 3 cont. ep.
    'bidirectional': True,
    'use_embeddings': True,  # supercedes `generate_word_embeddings`
    'lr_lower_bound': 1e-6,  # will only be used if use_scheduler is True
    'load_model_name': None,  # TODO: make this cli parameter
    'save_dir': 'saved_model/',
    'model_code': 'seq2seq',  # basic encoder-decoder framework w/ attn
    'vocab_path': 'data/vocab.pkl',
    'embeddings_path': 'embeddings/',
    'generate_word_embeddings': False,  # True only for the very first run
}


def get_dependent_hyperparams(modalities):
    """sets `enc_input_dim` and `pretrained_modality`"""
    # 43 for s2t, 2048 for v2t and =embedding_dim for t2t
    fusion_type = model_config['fusion']
    if fusion_type:
        if fusion_type == 'early':  # concatenate before passing through RNN
            input_dim = 0
            for m in modalities.split('-'):
                if m == 't':
                    input_dim += model_config['embedding_dim']
                elif m == 's':
                    input_dim += 43  # for how2 dataset
                elif m == 'v':
                    input_dim += 2048  # for how2 dataset
            model_config['enc_input_dim'] = input_dim
        elif fusion_type in ['late', 'tfn']:
            model_config['enc_input_dim'] = {
                's': 43, 'v': 2048, 't': model_config['embedding_dim']}
    elif model_config['modalities'] in ['ss-vv', 'v-s']:
        model_config['enc_input_dim'] = {'v': 2048, 's': 43,
                                         't': model_config['embedding_dim']}
        model_config['pretrained_modality'] = {
            'v': 'saved_model/seq2seq-vv-en-pt-0.pth',
            's': 'saved_model/seq2seq-ss-en-pt-0.pth',
            't': 'saved_model/seq2seq-t-en-pt-0.pth'}
    elif model_config['modalities'] == 't':
        model_config['enc_input_dim'] = model_config['embedding_dim']
    elif model_config['modalities'] in ['s', 'ss']:
        model_config['enc_input_dim'] = 43
    elif model_config['modalities'] in ['v', 'vv']:
        model_config['enc_input_dim'] = 2048
    return model_config


model_config = get_dependent_hyperparams(model_config['modalities'])
