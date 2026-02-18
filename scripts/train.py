import os
import sys
import pickle
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.efficient_phys import get_model, NegPearsonLoss
from utils.signal_processing import SignalProcessor
from utils.metrics import compute_all_metrics, print_metrics


# ---- Dataset Class ----
class RPPGDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, "rb") as f:
            data = pickle.load(f)

        self.clips = data["clips"]
        self.labels = data["labels"]

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip = self.clips[idx]
        label = self.labels[idx]

        frames = torch.FloatTensor(clip["frames"])    # (T, C, H, W)
        ppg = torch.FloatTensor(label)                 # (T,)

        return frames, ppg


# ---- Training Loop ----
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    n_batches = 0

    for batch_frames, batch_ppg in tqdm(dataloader, desc="Training", leave=False):
        batch_frames = batch_frames.to(device)
        batch_ppg = batch_ppg.to(device)

        # Forward
        pred_signal = model(batch_frames)  # (B, T-1)

        # Align lengths
        min_len = min(pred_signal.shape[1], batch_ppg.shape[1])
        pred_signal = pred_signal[:, :min_len]
        batch_ppg = batch_ppg[:, :min_len]

        loss = criterion(pred_signal, batch_ppg)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ---- Validation Loop ----
def validate(model, dataloader, criterion, device, signal_processor):
    model.eval()
    total_loss = 0
    n_batches = 0

    all_pred_hr = []
    all_true_hr = []

    with torch.no_grad():
        for batch_frames, batch_ppg in tqdm(dataloader, desc="Validating", leave=False):
            batch_frames = batch_frames.to(device)
            batch_ppg = batch_ppg.to(device)

            pred_signal = model(batch_frames)

            min_len = min(pred_signal.shape[1], batch_ppg.shape[1])
            pred_signal = pred_signal[:, :min_len]
            batch_ppg = batch_ppg[:, :min_len]

            loss = criterion(pred_signal, batch_ppg)
            total_loss += loss.item()
            n_batches += 1

            # Compute HR for each sample in batch
            for i in range(pred_signal.shape[0]):
                pred_np = pred_signal[i].cpu().numpy()
                true_np = batch_ppg[i].cpu().numpy()

                pred_hr = signal_processor.compute_heart_rate(pred_np)
                true_hr = signal_processor.compute_heart_rate(true_np)

                if pred_hr > 0 and true_hr > 0:
                    all_pred_hr.append(pred_hr)
                    all_true_hr.append(true_hr)

    avg_loss = total_loss / max(n_batches, 1)

    metrics = {}
    if all_pred_hr:
        metrics = compute_all_metrics(all_pred_hr, all_true_hr)

    return avg_loss, metrics


# ---- Main Training Function ----
def train():
    # Load config
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    tc = config["training"]
    device = torch.device(tc["device"])
    print(f"Device: {device}")

    # Create directories
    os.makedirs(tc["save_path"], exist_ok=True)
    os.makedirs(tc["log_path"], exist_ok=True)

    # Load data
    processed_path = config["dataset"]["processed_path"]
    train_dataset = RPPGDataset(os.path.join(processed_path, "train_data.pkl"))
    val_dataset = RPPGDataset(os.path.join(processed_path, "val_data.pkl"))

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=tc["batch_size"],
        shuffle=True,
        num_workers=tc["num_workers"],
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=tc["batch_size"],
        shuffle=False,
        num_workers=tc["num_workers"],
        pin_memory=False,
    )

    # Model
    model = get_model(
        img_size=config["model"]["img_size"],
        in_channels=config["model"]["input_channels"],
    )
    model = model.to(device)

    # Loss, Optimizer, Scheduler
    criterion = NegPearsonLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=tc["learning_rate"],
        weight_decay=tc["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=tc["scheduler_step"],
        gamma=tc["scheduler_gamma"],
    )

    signal_processor = SignalProcessor(fps=config["dataset"]["frame_rate"])

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_mae": [],
        "val_rmse": [],
    }

    best_val_loss = float("inf")

    # Training loop
    print(f"\nStarting training for {tc['epochs']} epochs...")
    print("=" * 60)

    for epoch in range(1, tc["epochs"] + 1):
        print(f"\nEpoch {epoch}/{tc['epochs']}")
        print("-" * 40)

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, device, signal_processor)

        # Step scheduler
        scheduler.step()

        # Log
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_metrics.get("MAE", 0))
        history["val_rmse"].append(val_metrics.get("RMSE", 0))

        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        if val_metrics:
            print(f"  Val MAE:    {val_metrics.get('MAE', 'N/A')} BPM")
            print(f"  Val RMSE:   {val_metrics.get('RMSE', 'N/A')} BPM")
            print(f"  Val Pearson:{val_metrics.get('Pearson', 'N/A')}")
        print(f"  LR:         {scheduler.get_last_lr()[0]:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(tc["save_path"], "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_metrics": val_metrics,
            }, save_path)
            print(f"  ✅ Best model saved! (loss: {val_loss:.4f})")

    # Save final model
    save_path = os.path.join(tc["save_path"], "final_model.pth")
    torch.save({
        "epoch": tc["epochs"],
        "model_state_dict": model.state_dict(),
        "val_loss": val_loss,
    }, save_path)

    # Plot training curves
    plot_training_curves(history, config["evaluation"]["plot_path"])

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved to: {tc['save_path']}")


def plot_training_curves(history, plot_path):
    """Plot and save training curves."""
    os.makedirs(plot_path, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss curves
    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True)

    # MAE
    axes[1].plot(history["val_mae"], label="Val MAE", color="orange")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE (BPM)")
    axes[1].set_title("Heart Rate MAE")
    axes[1].legend()
    axes[1].grid(True)

    # RMSE
    axes[2].plot(history["val_rmse"], label="Val RMSE", color="red")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("RMSE (BPM)")
    axes[2].set_title("Heart Rate RMSE")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_path, "training_curves.png"), dpi=150)
    plt.close()
    print(f"\nTraining curves saved to: {plot_path}/training_curves.png")


if __name__ == "__main__":
    train()