import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from datasets import load_dataset
from dataset import AudioDataset
from model import AudioCNN
import wandb

# --------- CONFIGURATIONS ---------
default_wandb_config = {
    "BATCH_SIZE": 32,
    "EPOCHS": 10,
    "LEARNING_RATE": 0.001,
    "DEVICE": "cpu",
    "FIXED_WIDTH": 751,
    "DATASET_NAME": "danavery/urbansound8K"
}

wandb.init(project="audio-classification", entity="your-entity", config=default_wandb_config)
CONFIG = wandb.config

# --- CONFIG PARAMS ---
BATCH_SIZE = CONFIG.BATCH_SIZE
EPOCHS = CONFIG.EPOCHS
LEARNING_RATE = CONFIG.LEARNING_RATE
DEVICE = CONFIG.DEVICE
FIXED_WIDTH = CONFIG.FIXED_WIDTH
DATASET_NAME = CONFIG.DATASET_NAME

# --------- DATASET PREPARATION ---------
class PaddedDataset(AudioDataset):
    def __getitem__(self, idx):
        spectrogram, label = super().__getitem__(idx)
        width = spectrogram.shape[-1]
        if width < FIXED_WIDTH:
            spectrogram = torch.nn.functional.pad(spectrogram, (0, FIXED_WIDTH - width))
        else:
            spectrogram = spectrogram[:, :, :FIXED_WIDTH]
        return spectrogram, label

# --------- TRAINING AND EVALUATION ---------
def train():
    dataset = load_dataset(DATASET_NAME, split="train")
    label_map = {label: idx for idx, label in enumerate(sorted(set(dataset["class"])))}

    train_len, val_len = int(0.8 * len(dataset)), int(0.1 * len(dataset))
    test_len = len(dataset) - train_len - val_len

    train_data, val_data, test_data = random_split(dataset, [train_len, val_len, test_len])

    train_loader = DataLoader(PaddedDataset(train_data, label_map), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(PaddedDataset(val_data, label_map), batch_size=BATCH_SIZE, num_workers=4)
    test_loader = DataLoader(PaddedDataset(test_data, label_map), batch_size=BATCH_SIZE, num_workers=4)

    model = AudioCNN(num_classes=len(label_map)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct = 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()

        train_loss = total_loss / len(train_loader.dataset)
        train_acc = correct / len(train_loader.dataset)

        model.eval()
        val_correct = sum((model(i.to(DEVICE)).argmax(1) == l.to(DEVICE)).sum().item() for i, l in val_loader)
        val_acc = val_correct / len(val_loader.dataset)

        wandb.log({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_acc": val_acc})
        print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

    test_correct = sum((model(i.to(DEVICE)).argmax(1) == l.to(DEVICE)).sum().item() for i, l in test_loader)
    test_acc = test_correct / len(test_loader.dataset)

    wandb.log({"test_acc": test_acc})
    print(f"Test Accuracy: {test_acc:.4f}")

    wandb.finish()
