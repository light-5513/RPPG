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
    """Run real-time rPPG demo with vitals monitoring."""
    print("=" * 60)
    print("🫀 Real-time rPPG Vitals Monitor")
    print("=" * 60)

    # Load configuration
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize components
    print("\n📦 Loading components...")
    model, device = load_model(config)
    face_extractor = FaceROIExtractor(roi_size=config["model"]["img_size"])
    signal_processor = SignalProcessor(fps=config["live_demo"]["frame_rate"])

    # Initialize camera
    camera_source = config["live_demo"]["camera_source"]
    print(f"📹 Connecting to camera: {camera_source}")
    
    cap = cv2.VideoCapture(camera_source)
    
    # If DroidCam URL fails, try webcam
    if not cap.isOpened():
        print("⚠️  DroidCam URL failed, trying webcam index 0...")
        camera_source = 0
        cap = cv2.VideoCapture(camera_source)
    
    if not cap.isOpened():
        print("❌ Could not open camera!")
        return

    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config["live_demo"]["display_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config["live_demo"]["display_height"])

    print("✅ Camera connected successfully")
    print(f"Buffer: {config['live_demo']['buffer_seconds']} seconds")
    print(f"Frame rate: {config['live_demo']['frame_rate']} fps")
    print("\nPress 'q' to quit\n")

    # Buffers for frames and signals
    buffer_size = config["live_demo"]["buffer_seconds"] * config["live_demo"]["frame_rate"]
    frame_buffer = deque(maxlen=buffer_size)
    rgb_buffer = deque(maxlen=buffer_size)
    rppg_signal_buffer = deque(maxlen=buffer_size)

    vitals = {}
    frame_count = 0
    last_inference_time = time.time()
    inference_interval = 1.0  # Run inference every 1 second

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️  Failed to read frame")
            break

        frame_count += 1
        display_frame = frame.copy()

        # Extract face ROI
        face_crop, rgb_mean, bbox = face_extractor.extract_face_roi(frame)

        if face_crop is not None and rgb_mean is not None:
            # Add to buffers
            frame_buffer.append(face_crop)
            rgb_buffer.append(rgb_mean)

            # Draw face bounding box
            x1, y1, x2, y2 = bbox
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Calculate buffer fill percentage
            buffer_percent = (len(frame_buffer) / buffer_size) * 100

            # Run inference if enough data and time elapsed
            min_frames = int(3 * config["live_demo"]["frame_rate"])  # At least 3 seconds
            current_time = time.time()

            if len(frame_buffer) >= min_frames and (current_time - last_inference_time) >= inference_interval:
                last_inference_time = current_time

                try:
                    # Prepare frames for model
                    frames_np = np.array(list(frame_buffer))  # (T, H, W, 3)
                    
                    # Convert to torch tensor and normalize
                    frames_tensor = torch.from_numpy(frames_np).float()
                    frames_tensor = frames_tensor.permute(0, 3, 1, 2)  # (T, 3, H, W)
                    frames_tensor = frames_tensor / 255.0

                    # Add batch dimension
                    frames_tensor = frames_tensor.unsqueeze(0)  # (1, T, 3, H, W)

                    # Run model inference
                    with torch.no_grad():
                        predicted_signal = model(frames_tensor)  # (1, T)
                        predicted_signal = predicted_signal.squeeze().cpu().numpy()

                    # Update rPPG signal buffer
                    for sig_val in predicted_signal:
                        rppg_signal_buffer.append(sig_val)

                    # Extract vitals
                    rppg_array = np.array(list(rppg_signal_buffer))
                    rgb_array = np.array(list(rgb_buffer))
                    
                    vitals = signal_processor.extract_all_vitals(rppg_array, rgb_array)

                except Exception as e:
                    print(f"⚠️  Inference error: {e}")
                    vitals = {}

            # Draw vitals panel
            display_frame = draw_vitals(display_frame, vitals, buffer_percent)

            # Draw signal plot
            if len(rppg_signal_buffer) > 0:
                display_frame = draw_signal_plot(display_frame, list(rppg_signal_buffer), 
                                                 fps=config["live_demo"]["frame_rate"])

        else:
            # No face detected
            h, w = display_frame.shape[:2]
            cv2.putText(display_frame, "No face detected", (w // 2 - 150, h // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # Display frame
        cv2.imshow("rPPG Live Demo - Press 'q' to quit", display_frame)

        # Check for quit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    print("\n🛑 Shutting down...")
    cap.release()
    cv2.destroyAllWindows()
    face_extractor.close()
    print("✅ Demo ended successfully")


if __name__ == "__main__":
    run_live_demo()