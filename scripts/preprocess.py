import os
import sys
import cv2
import numpy as np
import yaml
from tqdm import tqdm
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.face_detection import FaceROIExtractor


def load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)


def load_ground_truth(gt_path):
    """Load ground truth PPG signal from UBFC-rPPG format."""
    gt_data = np.loadtxt(gt_path)

    # UBFC-rPPG ground_truth.txt format (each row is a channel, columns are time):
    # Row 0: PPG signal (from pulse oximeter)
    # Row 1: Heart rate
    # Row 2: Timestamps (sometimes)

    if gt_data.ndim == 1:
        ppg_signal = gt_data
        hr_values = None
    else:
        # Data is (num_channels, num_frames) — rows are channels
        if gt_data.shape[0] <= 5 and gt_data.shape[1] > gt_data.shape[0]:
            # Shape like (3, N): rows are channels, columns are time points
            ppg_signal = gt_data[0, :]
            hr_values = gt_data[1, :] if gt_data.shape[0] > 1 else None
        else:
            # Shape like (N, 3): rows are time points, columns are channels
            ppg_signal = gt_data[:, 0]
            hr_values = gt_data[:, 1] if gt_data.shape[1] > 1 else None

    return ppg_signal, hr_values


def process_video(video_path, gt_path, face_extractor, config):
    """Process a single video into clips."""
    clip_length = config["preprocess"]["clip_length"]
    stride = config["preprocess"]["stride"]
    roi_size = config["preprocess"]["face_roi_size"]

    # Load ground truth
    ppg_signal, hr_values = load_ground_truth(gt_path)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: Cannot open video: {video_path}")
        return [], []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  Video: {total_frames} frames, {fps:.1f} FPS")

    # Read all frames and extract face ROIs
    frames = []
    rgb_signals = []

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        face_crop, rgb_mean, _ = face_extractor.extract_face_roi(frame)

        if face_crop is not None:
            # Normalize to [0, 1]
            face_normalized = face_crop.astype(np.float32) / 255.0
            frames.append(face_normalized)
            rgb_signals.append(rgb_mean)
        else:
            # Use previous frame if face not detected
            if frames:
                frames.append(frames[-1])
                rgb_signals.append(rgb_signals[-1])

    cap.release()

    if not frames:
        print(f"  ERROR: No faces detected in video")
        return [], []

    frames = np.array(frames)
    rgb_signals = np.array(rgb_signals)

    # Align PPG signal length with video frames
    min_len = min(len(frames), len(ppg_signal))
    frames = frames[:min_len]
    ppg_signal = ppg_signal[:min_len]
    rgb_signals = rgb_signals[:min_len]

    # Normalize PPG signal
    ppg_signal = (ppg_signal - ppg_signal.mean()) / (ppg_signal.std() + 1e-8)

    # Create clips
    clips = []
    labels = []

    for start in range(0, min_len - clip_length, stride):
        end = start + clip_length

        clip_frames = frames[start:end]          # (T, H, W, 3)
        clip_ppg = ppg_signal[start:end]          # (T,)
        clip_rgb = rgb_signals[start:end]          # (T, 3)

        # Transpose frames to (T, C, H, W)
        clip_frames = clip_frames.transpose(0, 3, 1, 2)

        clips.append({
            "frames": clip_frames,
            "rgb_signals": clip_rgb,
        })
        labels.append(clip_ppg)

    print(f"  Created {len(clips)} clips")
    return clips, labels


def preprocess_dataset():
    """Preprocess entire UBFC-rPPG dataset."""
    config = load_config()
    raw_path = config["dataset"]["raw_path"]
    processed_path = config["dataset"]["processed_path"]

    os.makedirs(processed_path, exist_ok=True)

    face_extractor = FaceROIExtractor(roi_size=config["preprocess"]["face_roi_size"])

    # Find all subjects
    subjects = sorted([
        d for d in os.listdir(raw_path)
        if os.path.isdir(os.path.join(raw_path, d))
    ])

    print(f"Found {len(subjects)} subjects")

    all_clips = []
    all_labels = []

    for subject in tqdm(subjects, desc="Processing subjects"):
        subject_path = os.path.join(raw_path, subject)

        # Find video and ground truth files
        video_path = None
        gt_path = None

        for f in os.listdir(subject_path):
            if f.endswith((".avi", ".mp4", ".mkv")):
                video_path = os.path.join(subject_path, f)
            if "ground_truth" in f.lower() or f.endswith(".txt"):
                gt_path = os.path.join(subject_path, f)

        if video_path is None or gt_path is None:
            print(f"  Skipping {subject}: missing video or ground truth")
            continue

        print(f"\nProcessing: {subject}")
        clips, labels = process_video(video_path, gt_path, face_extractor, config)
        all_clips.extend(clips)
        all_labels.extend(labels)

    face_extractor.close()

    # Split into train/val
    split_ratio = config["preprocess"]["train_split"]
    n_total = len(all_clips)
    n_train = int(n_total * split_ratio)

    # Shuffle
    indices = np.random.permutation(n_total)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_data = {
        "clips": [all_clips[i] for i in train_indices],
        "labels": [all_labels[i] for i in train_indices],
    }

    val_data = {
        "clips": [all_clips[i] for i in val_indices],
        "labels": [all_labels[i] for i in val_indices],
    }

    # Save processed data
    print(f"\nSaving processed data...")
    print(f"  Train: {len(train_data['clips'])} clips")
    print(f"  Val: {len(val_data['clips'])} clips")

    with open(os.path.join(processed_path, "train_data.pkl"), "wb") as f:
        pickle.dump(train_data, f)

    with open(os.path.join(processed_path, "val_data.pkl"), "wb") as f:
        pickle.dump(val_data, f)

    print(f"\nPreprocessing complete!")
    print(f"Saved to: {processed_path}")


if __name__ == "__main__":
    preprocess_dataset()