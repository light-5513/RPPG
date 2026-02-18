import os
import sys
import pickle
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.efficient_phys import get_model
from scripts.train import RPPGDataset
from utils.signal_processing import SignalProcessor
from utils.metrics import compute_all_metrics, print_metrics


def evaluate():
    # Load config
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(config["training"]["device"])
    plot_path = config["evaluation"]["plot_path"]
    os.makedirs(plot_path, exist_ok=True)

    # Load model
    model = get_model(
        img_size=config["model"]["img_size"],
        in_channels=config["model"]["input_channels"],
    )

    checkpoint_path = os.path.join(config["training"]["save_path"], "best_model.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Validation loss at save: {checkpoint['val_loss']:.4f}")

    # Load validation data
    processed_path = config["dataset"]["processed_path"]
    val_dataset = RPPGDataset(os.path.join(processed_path, "val_data.pkl"))
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    signal_processor = SignalProcessor(fps=config["dataset"]["frame_rate"])

    # Evaluate
    all_pred_hr = []
    all_true_hr = []
    all_pred_signals = []
    all_true_signals = []

    print("\nEvaluating...")
    with torch.no_grad():
        for batch_frames, batch_ppg in tqdm(val_loader, desc="Evaluating"):
            batch_frames = batch_frames.to(device)
            pred_signal = model(batch_frames)

            pred_np = pred_signal[0].cpu().numpy()
            true_np = batch_ppg[0].numpy()

            # Align lengths
            min_len = min(len(pred_np), len(true_np))
            pred_np = pred_np[:min_len]
            true_np = true_np[:min_len]

            # Compute HR
            pred_hr = signal_processor.compute_heart_rate(pred_np)
            true_hr = signal_processor.compute_heart_rate(true_np)

            if pred_hr > 0 and true_hr > 0:
                all_pred_hr.append(pred_hr)
                all_true_hr.append(true_hr)

            all_pred_signals.append(pred_np)
            all_true_signals.append(true_np)

    # Compute metrics
    if all_pred_hr:
        metrics = compute_all_metrics(all_pred_hr, all_true_hr)
        print_metrics(metrics, "Heart Rate Evaluation")

        # ---- Plot 1: Bland-Altman Plot ----
        pred_arr = np.array(all_pred_hr)
        true_arr = np.array(all_true_hr)
        mean_hr = (pred_arr + true_arr) / 2
        diff_hr = pred_arr - true_arr
        mean_diff = np.mean(diff_hr)
        std_diff = np.std(diff_hr)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].scatter(mean_hr, diff_hr, alpha=0.6, edgecolors="black", linewidths=0.5)
        axes[0].axhline(y=mean_diff, color="red", linestyle="-", label=f"Mean: {mean_diff:.2f}")
        axes[0].axhline(y=mean_diff + 1.96 * std_diff, color="gray", linestyle="--", label=f"+1.96 SD: {mean_diff + 1.96 * std_diff:.2f}")
        axes[0].axhline(y=mean_diff - 1.96 * std_diff, color="gray", linestyle="--", label=f"-1.96 SD: {mean_diff - 1.96 * std_diff:.2f}")
        axes[0].set_xlabel("Mean HR (BPM)")
        axes[0].set_ylabel("Difference (Pred - True) BPM")
        axes[0].set_title("Bland-Altman Plot")
        axes[0].legend()
        axes[0].grid(True)

        # ---- Plot 2: Scatter Plot ----
        axes[1].scatter(true_arr, pred_arr, alpha=0.6, edgecolors="black", linewidths=0.5)
        min_val = min(true_arr.min(), pred_arr.min()) - 5
        max_val = max(true_arr.max(), pred_arr.max()) + 5
        axes[1].plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect prediction")
        axes[1].set_xlabel("True HR (BPM)")
        axes[1].set_ylabel("Predicted HR (BPM)")
        axes[1].set_title(f"HR Prediction (Pearson: {metrics['Pearson']:.3f})")
        axes[1].legend()
        axes[1].grid(True)

        # ---- Plot 3: Sample Signal Comparison ----
        if all_pred_signals:
            sample_idx = 0
            pred_sig = signal_processor.bandpass_filter(all_pred_signals[sample_idx])
            true_sig = signal_processor.bandpass_filter(all_true_signals[sample_idx])
            time = np.arange(len(pred_sig)) / config["dataset"]["frame_rate"]

            axes[2].plot(time, true_sig / (np.max(np.abs(true_sig)) + 1e-8), label="Ground Truth", alpha=0.8)
            axes[2].plot(time, pred_sig / (np.max(np.abs(pred_sig)) + 1e-8), label="Predicted", alpha=0.8)
            axes[2].set_xlabel("Time (s)")
            axes[2].set_ylabel("Normalized Amplitude")
            axes[2].set_title("Sample rPPG Signal")
            axes[2].legend()
            axes[2].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, "evaluation_results.png"), dpi=150)
        plt.close()
        print(f"Plots saved to: {plot_path}/evaluation_results.png")

    else:
        print("No valid HR predictions found!")


if __name__ == "__main__":
    evaluate()