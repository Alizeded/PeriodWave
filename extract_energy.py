import math
import os
import random
import torch
import torch.utils.data
import numpy as np
from librosa.util import normalize
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn
import pathlib
from tqdm import tqdm

MAX_WAV_VALUE = 32768.0
def load_wav(full_path, sr_target):
    sampling_rate, data = read(full_path)
    if sampling_rate != sr_target:
        raise RuntimeError("Sampling rate of the file {} is {} Hz, but the model requires {} Hz".
              format(full_path, sampling_rate, sr_target))
    return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    # complex tensor as default, then use view_as_real for future pytorch compatibility
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec

def parse_filelist(filelist_path):
    with open(filelist_path, 'r') as f:
        filelist = [line.strip() for line in f.readlines()]
    return filelist
    

audio_files = parse_filelist("filelists_24k/train_wav.txt")

energy_list = []

print("INFO: computing training set waveform statistics for PriorGrad training...")

for i in tqdm(range(len(audio_files))):
    filename = audio_files[i]
    audio, sr = load_wav(filename, 24000)
    if 24000 != sr:
        raise ValueError(f'Invalid sample rate {sr}.')
    audio = audio / MAX_WAV_VALUE
    audio = normalize(audio) * 0.95

    audio = torch.FloatTensor(audio).cuda()
    audio = audio.unsqueeze(0)
    # match audio length to self.hop_size * n for evaluation
    if (audio.size(1) % 256) != 0:
        audio = audio[:, :-(audio.size(1) % 256)]

    mel = mel_spectrogram(audio, 1024, 100,
                            24000, 256, 1024, 0, 12000,
                            center=False)
    assert audio.shape[1] == mel.shape[2] * 256, "audio shape {} mel shape {}".format(audio.shape, mel.shape)

    energy = (mel.exp()).sum(1).sqrt()


    energy_list.append(energy.squeeze(0))



energy_list = torch.cat(energy_list)


energy_max = energy_list.max().cpu().numpy()
energy_min = energy_list.min().cpu().numpy()
os.makedirs("stats_libritts", exist_ok=True)
print("INFO: stats computed: max energy {} min energy {}".format(energy_max, energy_min))
np.save("stats_libritts/energy_max_train.npy", energy_max)
np.save("stats_libritts/energy_min_train.npy", energy_min)

