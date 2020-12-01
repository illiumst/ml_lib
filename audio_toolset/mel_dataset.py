from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


class TorchMelDataset(Dataset):
    def __init__(self, identifier, mel_path, segment_len, hop_len, label, padding=0, transform=None):
        self.padding = padding
        self.path = next(iter(Path(mel_path).glob(f'{identifier}_*')))
        self.segment_len = segment_len
        self.m, self.n = str(self.path).split('_')[-2:]  # get spectrogram dimensions
        self.n = int(self.n.split('.', 1)[0])  # remove .npy
        self.m, self.n = (int(i) for i in (self.m, self.n))
        self.offsets = list(range(0, self.n - segment_len, hop_len))
        self.label = label
        self.transform = transform

    def __getitem__(self, item):
        start = self.offsets[item]
        mel_spec = np.load(str(self.path), allow_pickle=True)
        if self.padding > 0:
            mel_spec = np.pad(mel_spec, pad_width=[(0, 0), (self.padding // 2, self.padding // 2)], mode='mean')
        snippet = mel_spec[:, start: start + self.segment_len]
        if self.transform:
            snippet = self.transform(snippet)
        return snippet, self.label

    def __len__(self):
        return len(self.offsets)