import time
from pathlib import Path

import pickle
from torch.utils.data import Dataset

from ml_lib.modules.util import AutoPadToShape


class TorchMelDataset(Dataset):
    def __init__(self, mel_path, sub_segment_len, sub_segment_hop_len, label, audio_file_len,
                 sampling_rate, mel_hop_len, n_mels, transform=None, auto_pad_to_shape=True):
        super(TorchMelDataset, self).__init__()
        self.sampling_rate = sampling_rate
        self.audio_file_len = audio_file_len
        self.padding = AutoPadToShape((n_mels , sub_segment_len)) if auto_pad_to_shape else None
        self.path = Path(mel_path)
        self.sub_segment_len = sub_segment_len
        self.mel_hop_len = mel_hop_len
        self.sub_segment_hop_len = sub_segment_hop_len
        self.n = int((self.sampling_rate / self.mel_hop_len) * self.audio_file_len + 1)
        self.offsets = list(range(0, self.n - self.sub_segment_len, self.sub_segment_hop_len))
        self.label = label
        self.transform = transform

    def __getitem__(self, item):
        while Path(str(self.path).replace(self.path.suffix, '.lock')).exists():
            time.sleep(0.01)
        with self.path.open('rb') as mel_file:
            mel_spec = pickle.load(mel_file, fix_imports=True)
        start = self.offsets[item]
        snippet = mel_spec[: , start: start + self.sub_segment_len]
        if self.transform:
            snippet = self.transform(snippet)
        if self.padding:
            snippet = self.padding(snippet)
        return snippet, self.label

    def __len__(self):
        return len(self.offsets)
