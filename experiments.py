
import torchaudio
if sys.platform =='windows':
    torchaudio.set_audio_backend('soundfile')
else:
    torchaudio.set_audio_backend('sox_io')


class PyTorchAudioToMelDataset(_AudioToMelDataset):

    @property
    def audio_file_duration(self):
        info_obj = torchaudio.info(self.audio_path)
        return info_obj.num_frames / info_obj.sample_rate

    @property
    def sampling_rate(self):
        return self.mel_kwargs['sample_rate']

    def __init__(self, audio_file_path, *args, **kwargs):
        super(PyTorchAudioToMelDataset, self).__init__(audio_file_path, *args, **kwargs)

        audio_file_path = Path(audio_file_path)
        # audio_file, sampling_rate = librosa.load(self.audio_path, sr=sampling_rate)

        from torchaudio.transforms import MelSpectrogram
        self._mel_transform = Compose([MelSpectrogram(**self.mel_kwargs),
                                       MelToImage()
                                       ])

    def _build_mel(self):
        if self.reset:
            self.mel_file_path.unlink(missing_ok=True)
        if not self.mel_file_path.exists():
            self.mel_file_path.parent.mkdir(parents=True, exist_ok=True)
            lock_file = Path(str(self.mel_file_path).replace(self.mel_file_path.suffix, '.lock'))
            lock_file.touch(exist_ok=False)

            try:
                audio_sample, sample_rate = torchaudio.load(self.audio_path)
            except RuntimeError:
                import soundfile

                data, samplerate = soundfile.read(self.audio_path)
                # sf.available_formats()
                # sf.available_subtypes()
                soundfile.write(self.audio_path, data, samplerate, subtype='PCM_32')

                audio_sample, sample_rate = torchaudio.load(self.audio_path)
            if sample_rate != self.sampling_rate:
                resample = torchaudio.transforms.Resample(orig_freq=int(sample_rate), new_freq=int(self.sampling_rate))
                audio_sample = resample(audio_sample)
            if audio_sample.shape[0] > 1:
                # Transform Stereo to Mono
                audio_sample = audio_sample.mean(dim=0, keepdim=True)
            mel_sample = self._mel_transform(audio_sample)
            with self.mel_file_path.open('wb') as mel_file:
                pickle.dump(mel_sample, mel_file, protocol=pickle.HIGHEST_PROTOCOL)
            lock_file.unlink()
        else:
            # print(f"Already existed.. Skipping {filename}")
            # mel_file = mel_file
            pass

        # with mel_file.open(mode='rb') as f:
        #     mel_sample = pickle.load(f, fix_imports=True)
        return self.mel_file_path.exists()
