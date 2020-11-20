# from https://github.com/NVIDIA/tacotron2
# modified load_wav_to_torch, resampling to match the target_sampling_rate

import numpy as np
from scipy.io.wavfile import read
import torch


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


def load_wav_to_torch(full_path, target_sampling_rate):
    sr, data = read(full_path)
    if sr != target_sampling_rate:
        data = librosa.resample(data.astype(np.float32), sr, target_sampling_rate)
    return torch.FloatTensor(data.astype(np.float32)), target_sampling_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)
