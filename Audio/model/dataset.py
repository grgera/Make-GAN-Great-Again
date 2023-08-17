import os, librosa, glob, shutil, torch, random
import numpy as np

def make_dir(name):
    os.makedirs(name, exist_ok=True)
    os.makedirs(name + '/mel', exist_ok=True)
    os.makedirs(name + '/audio', exist_ok=True)
    
def get_mel(y, h, task=None):
    wav = torch.nn.functional.pad(y, ((h.n_fft - h.hop_length) // 2, (h.n_fft - h.hop_length) // 2), 
                                  mode='reflect')
    
    if task == 'preprocess':
        wav = wav.squeeze(0).squeeze(0)
        h.device = 'cpu'
    elif task == 'train':
        wav = wav.squeeze(1)
        
    spec = torch.stft(wav, h.n_fft, h.hop_length, h.win_length, center=False, 
                      window=torch.hann_window(h.win_length).to(h.device))
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
    
    mel_filter = librosa.filters.mel(h.sample_rate, h.n_fft, h.mel_dim)
    mel_filter = torch.from_numpy(mel_filter).to(h.device)
    mel_spec = torch.matmul(mel_filter, spec)
    mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
    
    return mel_spec
    
class MelDataset(torch.utils.data.Dataset):
    def __init__(self, mel_list, audio_list, h):
        self.h = h
        self.mel_list = mel_list
        self.audio_list = audio_list

    def __len__(self):
        return len(self.mel_list)

    def __getitem__(self, idx):
        mel = np.load(self.mel_list[idx])
        mel = torch.from_numpy(mel).float()
        start = random.randint(0, mel.size(1) - self.h.seq_len - 1)
        mel = mel[:, start : start + self.h.seq_len]

        wav = np.load(self.audio_list[idx])
        wav = torch.from_numpy(wav).float()
        start *= self.h.hop_length
        wav = wav[start : start + self.h.seq_len * self.h.hop_length]

        return mel, wav.unsqueeze(0)