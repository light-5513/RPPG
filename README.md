# Remote Photoplethysmography (rPPG) - Heart Rate Estimation

A deep learning-based system for contactless heart rate monitoring using remote photoplethysmography (rPPG) with real-time video analysis.

## 📋 Overview

This project implements an rPPG system that estimates heart rate from video by detecting subtle color changes in facial skin caused by blood circulation. The system can work with both pre-recorded datasets and real-time camera feeds (including DroidCam for smartphone cameras).

## 🗂️ Project Structure

```
V1/
├── datasets/
│   └── ubfc-rppg/          # Downloaded dataset goes here
├── models/
│   └── saved/              # Trained models saved here
├── scripts/
│   ├── download_dataset.py # Kaggle download script
│   ├── preprocess.py       # Data preprocessing
│   ├── train.py            # Model training
│   ├── evaluate.py         # Model evaluation
│   └── live_demo.py        # Real-time demo with DroidCam
├── utils/
│   ├── signal_processing.py # Filters, FFT, peak detection
│   ├── face_detection.py    # Face ROI extraction
│   └── metrics.py           # MAE, RMSE, Pearson etc.
├── config/
│   └── config.yaml          # All settings in one place
├── outputs/
│   ├── logs/                # Training logs
│   └── plots/               # Graphs & visualizations
├── requirements.txt         # All dependencies
└── README.md                # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Webcam or smartphone with DroidCam (for live demo)
- GPU recommended for training (CUDA-compatible)

### Installation

1. **Clone the repository** (or navigate to the project directory)

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Kaggle API** (for dataset download):
   - Create a Kaggle account and generate API token
   - Place `kaggle.json` in `~/.kaggle/` directory

## 📊 Dataset

This project uses the **UBFC-rPPG** dataset containing synchronized video and ground truth PPG signals.

### Download Dataset

```bash
cd scripts
python download_dataset.py
```

## 🔧 Usage

### 1. Preprocess Data

```bash
cd scripts
python preprocess.py
```

This script:
- Loads video files
- Detects faces and extracts ROIs
- Saves preprocessed data for training

### 2. Train Model

```bash
python train.py
```

Training configuration can be modified in `config/config.yaml`:
- Batch size, learning rate, epochs
- Model architecture
- Signal processing parameters

### 3. Evaluate Model

```bash
python evaluate.py
```

Generates:
- Performance metrics (MAE, RMSE, Pearson correlation)
- Bland-Altman plots
- Error distribution visualizations

### 4. Run Live Demo

```bash
python live_demo.py
```

**For DroidCam:**
1. Install DroidCam on your smartphone
2. Note the IP address shown in the app
3. Modify `camera_source` in the script or `config.yaml`:
   ```python
   camera_source = "http://192.168.1.100:4747/video"
   ```

## ⚙️ Configuration

All settings are centralized in `config/config.yaml`:

- **Dataset parameters**: Paths, train/val/test splits
- **Signal processing**: Bandpass filter settings, FFT parameters
- **Model architecture**: Network layers, dimensions
- **Training**: Batch size, learning rate, optimizer
- **Face detection**: Method (MediaPipe, Haar, etc.)

## 📈 Metrics

The system evaluates performance using:

- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **Pearson Correlation Coefficient**
- **MAPE** (Mean Absolute Percentage Error)
- **Bland-Altman Analysis**

## 🛠️ Technical Details

### Signal Processing Pipeline

1. Face detection and ROI extraction (forehead/cheeks)
2. Mean RGB signal extraction per frame
3. Bandpass filtering (0.7-4.0 Hz for 42-240 BPM range)
4. Detrending
5. FFT-based heart rate estimation

### Model Architecture

- Input: RGB video frames with detected face ROI
- Processing: CNN-based feature extraction
- Output: Heart rate estimation in BPM

## 📝 Dependencies

Key libraries:
- **PyTorch**: Deep learning framework
- **OpenCV**: Video processing
- **MediaPipe**: Face detection
- **SciPy**: Signal processing
- **NumPy/Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Visualization

See `requirements.txt` for complete list.

## 🔍 Troubleshooting

**No face detected:**
- Ensure good lighting conditions
- Keep face clearly visible and frontal
- Try different face detection methods in config

**Low accuracy:**
- Check camera stability (minimize motion)
- Ensure sufficient video duration (>10 seconds)
- Verify lighting is consistent
- Retrain with more data

**Camera not found:**
- Check camera index (try 0, 1, 2...)
- For DroidCam, verify IP address and port
- Ensure DroidCam app is running

## 📚 References

- UBFC-rPPG Dataset
- Remote Photoplethysmography research papers
- MediaPipe Face Detection

## 📄 License

This project is for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## ✨ Future Enhancements

- [ ] Support for multiple faces
- [ ] Motion compensation algorithms
- [ ] Additional rPPG methods (CHROM, POS, ICA)
- [ ] Mobile app integration
- [ ] Real-time performance optimization
- [ ] Multi-vital sign estimation (respiration rate, HRV)

---

**Note:** This is a research/educational project. Not intended for medical diagnosis.
