#import dataset (urbansound8K)
from datasets import load_dataset
import soundfile as sf
import torch
from torchvision import transforms
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


dataset = load_dataset("danavery/urbansound8K")

print(dataset["train"][0])
audio = dataset["train"][0]["audio"]["array"]
sr = dataset["train"][0]["audio"]["sampling_rate"]
hop_length = 512
label = dataset["train"][0]["class"]
print(label)

plt.figure(figsize=(14, 4))
librosa.display.waveshow(audio, sr=sr)
plt.title(f"Waveform: {label}")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
# plt.show()


mel_signal = librosa.feature.melspectrogram(sr=sr, y =audio)
# spectrogram = np.abs(mel_signal)
power_to_db = librosa.power_to_db(mel_signal, ref=np.max)

print(power_to_db.shape)

plt.figure(figsize=(8, 7))
image =librosa.display.specshow(power_to_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
plt.colorbar()
plt.title('Mel-Spectrogram (dB)')
plt.xlabel('Time')
plt.ylabel('Frequency')
# plt.show()

print(mel_signal.shape)
print(power_to_db.shape)


# Create tensor and check shape

x = torch.tensor(power_to_db).unsqueeze(0).unsqueeze(0)
x = x.float()
print(f"Input tensor shape: {x.shape}")


class AudioCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(AudioCNN, self).__init__()
        
        # First conv block: 1 channel input, 32 filters, 3x3 kernels, padding=1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        # Second conv block: 32 input channels, 64 filters, 3x3 kernels, padding=1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        # Third conv block: 64 input channels, 128 filters, 3x3 kernels, padding=1
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.dropout = nn.Dropout(0.3)
        
        # Calculate flatten size dynamically
        self.flatten_size = None
        
        # Fully connected layers (will be initialized after calculating flatten_size)
        self.fc1 = None
        self.fc2 = None
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = F.relu(self.conv3(x))
        
        x = self.dropout(x)
        
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
model = AudioCNN(num_classes=10)
output = model(x)
print(f"Output shape: {output.shape}")

