# Some of the modules mentioned here are taken directly from:
# https://github.com/vesis84/kaldi-io-for-python/blob/master/kaldi_io/kaldi_io.py
import pickle
import kaldi_io


def read_scp_raw(text):
    """
    Parameters:
    ------------
    text: str
        formatted string in proper scp format

    Returns:
    ------------
    key: str
        ark file's identifier
    mat: array
        ark vector corresponding to the audio sample
    Example:
    -----------
    key, mat = read_scp_raw(44Qq77fK4ew_6 ARK_PATH/sample.ark:215510338)
    """
    key, rxfile = text.split()
    return key, kaldi_io.read_mat(rxfile)


def scp2dict():
    """Construct a dictionary with sample id as key & ark vector address as value
    """
    modes = ['train', 'dev5', 'val']
    scp_dir = 'data/s2t/fbank_pitch_181506/'
    for mode in modes:
        scp_path = '{}{}.feats.scp'.format(scp_dir, mode)
        mode_scp_dict = {}
        with open(scp_path) as f:
            lines = f.readlines()
        for l in lines:
            key, value = l.strip().split(' ', 1)
            # this file doesn't have ark file in supported format. so excluding
            if key in ['D0T7ho08Q3o_25', 'FUfPuPMxh2w_9']:
                continue
            mode_scp_dict[key] = value
        with open('data/s2t/{}.feat_map.pkl'.format(mode), 'wb') as f:
            pickle.dump(mode_scp_dict, f)
        print('Done with {} mode'.format(mode))


def get_ark_rep(key, feat_map):
    return feat_map.get(key)


if __name__ == '__main__':
    scp_raw = '44Qq77fK4ew_6 ARK_PATH/raw_fbank_pitch_all_181506.3.ark:215510338'
    # scp_raw = '44Qq77fK4ew_6 data/s2t/fbank_pitch_181506/raw_fbank_pitch_all_181506.3.ark:215510338'
    print(read_scp_raw(scp_raw))

    scp2dict()
