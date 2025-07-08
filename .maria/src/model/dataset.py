import torch
import torch.utils.data as data
import librosa
import numpy as np


N_MELS = 64
HOP_LENGTH = 512
FIXED_WIDTH = 751
TARGET_SR = 22050
OUTPUT_DIR = "mel_spectrograms"


class AudioDataset(data.Dataset):
    def __init__(self, hf_dataset, label_map, hop_length=512, n_mels=64):
      
        self.dataset = hf_dataset
        self.label_map = label_map
        self.hop_length = hop_length
        self.n_mels = n_mels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        audio = sample["audio"]["array"]
        sr = sample["audio"]["sampling_rate"]
        label_str = sample["class"]

        # Convert waveform to Mel spectrogram (power), then to dB scale
        mel_signal = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmax=sr//2  # Ensure mel bins use full frequency range

        )
        power_to_db = librosa.power_to_db(mel_signal, ref=np.max)

        # Convert to tensor and add channel dimension (for CNN input)
        spectrogram = torch.tensor(power_to_db).unsqueeze(0).float()

        # Convert string label to integer tensor
        label = torch.tensor(self.label_map[label_str])
        if spectrogram.shape[-1] < FIXED_WIDTH:
            spectrogram = torch.nn.functional.pad(spectrogram, (0, FIXED_WIDTH - spectrogram.shape[-1]))
        else:
            spectrogram = spectrogram[:, :, :FIXED_WIDTH]
        return spectrogram, label
    

    class SavedMelDataset(data.Dataset):
     def __init__(self, mel_dir, label_map):
        self.mel_files = sorted([os.path.join(mel_dir, f) for f in os.listdir(mel_dir) if f.endswith('.pt')])
        self.label_map = label_map

     def __len__(self):
        return len(self.mel_files)

     def __getitem__(self, idx):
        data = torch.load(self.mel_files[idx])
        spectrogram = data['spectrogram']
        label_str = data['label']
        label = torch.tensor(self.label_map[label_str])
        
        return spectrogram, label