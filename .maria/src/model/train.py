import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from datasets import load_dataset
from dataset import AudioDataset
from model import AudioCNN
from sweep_config import sweep_config
import wandb
import os


# --- Config and W&B ---
default_config = {
    "BATCH_SIZE": 2,
    "EPOCHS": 10,
    "LEARNING_RATE": 0.001,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "FIXED_WIDTH": 751,
    "DATASET_NAME": "danavery/urbansound8K",
    "DATASET_SPLIT": "train[:20%]",
    "WEIGHT_DECAY": 1e-4,
    "DROPOUT_RATE": 0.3
}

# --- Padded Dataset ---
class PaddedDataset(AudioDataset):
    def __init__(self, dataset, label_map, fixed_width):
        super().__init__(dataset, label_map)
        self.fixed_width = fixed_width

    def __getitem__(self, idx):
        spec, label = super().__getitem__(idx)
        width = spec.shape[-1]
        if width < self.fixed_width:
            spec = torch.nn.functional.pad(spec, (0, self.fixed_width - width))
        else:
            spec = spec[:, :, :self.fixed_width]
        return spec, label

# --- Helper Functions ---
def calculate_accuracy_and_loss(model, data_loader, loss_fn, device):
    """Calculate accuracy and loss for a given data loader"""
    model.eval()
    correct, total, loss_sum = 0, 0, 0
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss = loss_fn(out, y)
            loss_sum += loss.item() * X.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total, loss_sum / total

# --- Training Function ---
def train():
    # Initialize wandb for sweep runs
    if not wandb.run:
        wandb.init()
    
    config = wandb.config
    
    ds = load_dataset(config.DATASET_NAME, split=config.DATASET_SPLIT)
    labels = sorted(set(ds["class"]))
    label_map = {l: i for i, l in enumerate(labels)}
    
    train_len = int(0.7 * len(ds))
    val_len = int(0.15 * len(ds))
    test_len = len(ds) - train_len - val_len
    train_ds, val_ds, test_ds = random_split(ds, [train_len, val_len, test_len])

    def make_loader(split):
     return DataLoader(
        PaddedDataset(split, label_map, config.FIXED_WIDTH),  # <-- now correct
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=config.DEVICE == "cuda"
    )

    train_loader = make_loader(train_ds)
    val_loader = make_loader(val_ds)
    test_loader = make_loader(test_ds)

    model = AudioCNN(num_classes=len(label_map), dropout_rate=config.DROPOUT_RATE).to(config.DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    wandb.watch(model)
    best_loss, best_state, patience = float("inf"), None, 0

    for epoch in range(config.EPOCHS):
        model.train()
        correct, total, loss_sum = 0, 0, 0
        for X, y in train_loader:
            X, y = X.to(config.DEVICE), y.to(config.DEVICE)
            optimizer.zero_grad()
            out = model(X)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * X.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
        train_acc = correct / total
        train_loss = loss_sum / total

        # Validation
        val_acc, val_loss_avg = calculate_accuracy_and_loss(model, val_loader, loss_fn, config.DEVICE)

        wandb.log({
            "epoch": epoch+1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss_avg,
            "val_acc": val_acc
        })

        if val_loss_avg < best_loss:
            best_loss = val_loss_avg
            best_state = model.state_dict()
            patience = 0
        else:
            patience += 1
        if patience >= 3:
            break

    if best_state:
        model.load_state_dict(best_state)

    # --- Testing ---
    test_acc, _ = calculate_accuracy_and_loss(model, test_loader, loss_fn, config.DEVICE)
    wandb.log({"test_acc": test_acc})

# --- Entry Point ---
if __name__ == "__main__":
    run_type = "train"  # or "sweep"
    print("RUNNING TYPE:", run_type)

    if run_type == "train":
        wandb.init(
            project="audio-classification-task1",
            entity="attp-ml-institute",
            config=default_config,
            name=f"audio-cnn-bs{default_config['BATCH_SIZE']}-lr{default_config['LEARNING_RATE']}"
        )
        train()

    elif run_type == "sweep":
        sweep_id = wandb.sweep(sweep_config, project="audio-classification-task1")
        wandb.agent(sweep_id=sweep_id, function=train)

    wandb.finish()
