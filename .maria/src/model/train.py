import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
import numpy as np
from dataset import AudioDataset  # your custom dataset class
from model import AudioCNN         # your model class

# Load dataset split
dataset = load_dataset("danavery/urbansound8K", split="train")

# Create label mapping
all_labels = sorted(set(dataset["class"]))
label_map = {label: idx for idx, label in enumerate(all_labels)}

# Define fixed width from your 95th percentile calculation
FIXED_WIDTH = 751

# Modify your AudioDataset to accept fixed width for padding/truncation
class PaddedAudioDataset(AudioDataset):
    def __init__(self, hf_dataset, label_map, fixed_width=FIXED_WIDTH, hop_length=512, n_mels=128):
        super().__init__(hf_dataset, label_map, hop_length, n_mels)
        self.fixed_width = fixed_width

    def __getitem__(self, idx):
        spectrogram, label = super().__getitem__(idx)
        width = spectrogram.shape[-1]
        if width < self.fixed_width:
            pad_amount = self.fixed_width - width
            spectrogram = torch.nn.functional.pad(spectrogram, (0, pad_amount))
        else:
            spectrogram = spectrogram[:, :, :self.fixed_width]
        return spectrogram, label

# Create dataset and dataloader
train_dataset = PaddedAudioDataset(dataset, label_map)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# Instantiate model
model = AudioCNN(num_classes=len(label_map), input_shape=(1, 128, 751))  # Set input shape to match your spectrograms
device = torch.device("cpu")
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f}")
