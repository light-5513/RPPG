import os
import sys
import cv2
import numpy as np
import torch
import yaml
import time
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.efficient_phys import get_model
from utils.face_detection import FaceROIExtractor
from utils.signal_processing import SignalProcessor


def load_model(config):
    """Load trained model."""
    device = torch.device("cpu")

    model = get_model(
        img_size=config["model"]["img_size"],
        in_channels=config["model"]["input_channels"],
    )

    checkpoint_path = os.path.join(config["training"]["save_path"], "best_model.pth")

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("✅ Loaded trained model")
    else:
        print("⚠️  No trained model found! Using untrained model (results will be random)")

    model.eval()
    return model, device


def draw_vitals(frame, vitals, buffer_percent):
    """Draw vital signs on frame."""
    h, w = frame.shape[:2]

    # Background panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (w - 320, 0), (w, 250), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    x = w - 310
    color_good = (0, 255, 0)
    color_warn = (0, 255, 255)
    color_bad = (0, 0, 255)
    white = (255, 255, 255)

    # Title
    cv2.putText(frame, "rPPG Vitals", (x, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, white, 2)

    # Heart Rate
    hr = vitals.get("heart_rate_bpm", 0)
    hr_color = color_good if 50 < hr < 120 else color_warn if hr > 0 else color_bad
    cv2.putText(frame, f"HR: {hr:.0f} BPM", (x, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, hr_color, 2)

    # SpO2
    spo2 = vitals.get("spo2_percent", 0)
    spo2_color = color_good if spo2 >= 95 else color_warn if spo2 >= 90 else color_bad
    cv2.putText(frame, f"SpO2: {spo2:.1f}%", (x, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, spo2_color, 2)

    # Respiratory Rate
    rr = vitals.get("respiratory_rate_bpm", 0)
    rr_color = color_good if 10 < rr < 25 else color_warn
    cv2.putText(frame, f"Resp: {rr:.0f} br/min", (x, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, rr_color, 2)

    # HRV
    hrv = vitals.get("hrv", {})
    sdnn = hrv.get("SDNN", 0)
    rmssd = hrv.get("RMSSD", 0)
    cv2.putText(frame, f"HRV SDNN: {sdnn:.1f} ms", (x, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.6, white, 1)
    cv2.putText(frame, f"HRV RMSSD: {rmssd:.1f} ms", (x, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, white, 1)

    # Signal Quality
    snr = vitals.get("signal_quality_snr_db", 0)
    quality = "Good" if snr > 5 else "Fair" if snr > 2 else "Poor"
    q_color = color_good if snr > 5 else color_warn if snr > 2 else color_bad
    cv2.putText(frame, f"Quality: {quality} ({snr:.1f}dB)", (x, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, q_color, 1)

    # Buffer progress bar
    bar_width = 200
    bar_height = 15
    bar_x = 10
    bar_y = h - 30
    filled = int(bar_width * buffer_percent / 100)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_height), (0, 200, 0), -1)
    cv2.putText(frame, f"Buffer: {buffer_percent:.0f}%", (bar_x + bar_width + 10, bar_y + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 1)

    return frame


def draw_signal_plot(frame, signal_buffer, fps=30):
    """Draw rPPG signal waveform at bottom of frame."""
    h, w = frame.shape[:2]

    if len(signal_buffer) < 10:
        return frame

    signal = np.array(signal_buffer)

    # Normalize for display
    sig_min = signal.min()
    sig_max = signal.max()
    if sig_max - sig_min > 0:
        signal_norm = (signal - sig_min) / (sig_max - sig_min)
    else:
        signal_norm = signal * 0

    # Draw area
    plot_h = 80
    plot_y = h - 60 - plot_h
    plot_x = 10
    plot_w = w - 340

    # Background
    overlay = frame.copy()
    cv2.rectangle(overlay, (plot_x - 5, plot_y - 5), (plot_x + plot_w + 5, plot_y + plot_h + 5), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Draw signal
    n_points = len(signal_norm)
    for i in range(1, n_points):
        x1 = plot_x + int((i - 1) / n_points * plot_w)
        x2 = plot_x + int(i / n_points * plot_w)
        y1 = plot_y + plot_h - int(signal_norm[i - 1] * plot_h)
        y2 = plot_y + plot_h - int(signal_norm[i] * plot_h)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(frame, "rPPG Signal", (plot_x, plot_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame


def run_live_demo():
    """Run real-time rP"""