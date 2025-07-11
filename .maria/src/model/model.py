import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import torch.utils.data as data
import numpy as np


class AudioCNN(nn.Module):
    def __init__(self, num_classes=10, in_channels=1, dropout_rate=0.3):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.dropout(x)
        x = self.pool(x)          # (batch, 128, 1, 1)
        x = x.view(x.size(0), -1) # (batch, 128)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x