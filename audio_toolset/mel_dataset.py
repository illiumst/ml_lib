import time
from pathlib import Path

import pickle
from torch.utils.data import Dataset

from ml_lib.modules.util import AutoPadToShape


class TorchMelDataset(Dataset):
    def __init__(self, mel_path, sub_segment_len, sub_segment_hop_len, label, audio_file_len,
                 sampling_rate, mel_hop_len, n_mels, transform=None, auto_pad_to_shape=True):
        super(TorchMelDataset, self).__init__()
        self.sampling_rate = int(sampling_rate)
        self.audio_file_len = float(audio_file_len)
        if auto_pad_to_shape and sub_segment_len:
            self.padding = AutoPadToShape((int(n_mels), int(sub_segment_len)))
        else:
            self.padding = None
        self.path = Path(mel_path)
        self.sub_segment_len = int(sub_segment_len)
        self.mel_hop_len = int(mel_hop_len)
        self.sub_segment_hop_len = int(sub_segment_hop_len)
        self.n = int((self.sampling_rate / self.mel_hop_len) * self.audio_file_len + 1)
        if self.sub_segment_len and self.sub_segment_hop_len and (self.n - self.sub_segment_len) > 0:
            self.offsets = list(range(0, self.n - self.sub_segment_len, self.sub_segment_hop_len))
        else:
            self.offsets = [0]
        if len(self) == 0:
            print('what happend here')
        self.label = label
        self.transform = transform

    def __getitem__(self, item):
        with self.path.open('rb') as mel_file:
            mel_spec = pickle.load(mel_file, fix_imports=True)
        start = self.offsets[item]
        sub_segments_attributes_set = self.sub_segment_len and self.sub_segment_hop_len
        sub_segment_length_smaller_then_tot_length = self.sub_segment_len < mel_spec.shape[1]

        if sub_segments_attributes_set and sub_segment_length_smaller_then_tot_length:
            duration = self.sub_segment_len
        else:
            duration = mel_spec.shape[1]

        snippet = mel_spec[:, start: start + duration]
        if self.transform:
            snippet = self.transform(snippet)
        if self.padding:
            snippet = self.padding(snippet)
        return self.path.__str__(), snippet, self.label

    def __len__(self):
        return len(self.offsets)
