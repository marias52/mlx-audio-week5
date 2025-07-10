import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioCNN(nn.Module):
    def __init__(self, num_classes=10, input_shape=(1, 128, 751)):
        super().__init__()

        # Define convolutional blocks
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
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
        self.dropout = nn.Dropout(0.3)

        # Dynamically calculate flatten size
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)  # (batch=1, channel=1, height=128, width=751)
            dummy = self.block1(dummy)
            dummy = self.block2(dummy)
            dummy = self.block3(dummy)
            self.flatten_size = dummy.numel() // dummy.shape[0]

        self.fc1 = nn.Linear(self.flatten_size, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.dropout(x)
        print(f"Shape before flattening: {x.shape}")  # Add this line

        x = torch.flatten(x, 1)  # Flatten all except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x