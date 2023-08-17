import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import argparse
import json
from utils import AttrDict
from tqdm import tqdm
from dataset import *

def preprocess(a, h):
    metadata = pd.read_csv(a.input_metadata_file, dtype='object', sep='|', header=None)
    wav_dir = metadata[0].values
    
    make_dir(h.out_dir)
    make_dir(h.valid_dir)
    
    for idx, fn in enumerate(tqdm(wav_dir)):
        file_dir = a.input_wavs_dir + fn + ".wav"
        wav, _ = librosa.load(file_dir, sr=h.sample_rate)
    
        wav = torch.from_numpy(wav)
        wav = wav.unsqueeze(0).unsqueeze(0)
        mel_spec = get_mel(wav, h, 'preprocess').numpy()

        mel_name = h.mel_name % idx
        np.save(os.path.join(h.out_dir + '/mel', mel_name), mel_spec, allow_pickle=False)

        audio_name = h.audio_name % idx
        np.save(os.path.join(h.out_dir + '/audio', audio_name), wav.squeeze(0).squeeze(0), allow_pickle=False)
    
    mel_list = sorted(glob.glob(os.path.join(h.out_dir + '/mel', '*.npy')))
    wav_list = sorted(glob.glob(os.path.join(h.out_dir + '/audio', '*.npy')))
    
    for i in range(h.valid_n):
        shutil.move(mel_list[i], h.valid_dir + '/mel')
        shutil.move(wav_list[i], h.valid_dir + '/audio')
    

def main():
    print('Building data...')

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_wavs_dir', default='./LJSpeech-1.1/wavs/')
    parser.add_argument('--input_metadata_file', default='./LJSpeech-1.1/metadata.csv')
    parser.add_argument('--config', default='')

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        h.device = 'cuda'
    else:
        h.device = 'cpu'

    preprocess(a, h)

if __name__ == '__main__':
    main()