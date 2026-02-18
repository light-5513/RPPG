import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, welch
from scipy.fft import fft, fftfreq


class SignalProcessor:
    """Process rPPG signals to extract vitals."""

    def __init__(self, fps=30):
        self.fps = fps

    # ---- Bandpass Filter ----
    def bandpass_filter(self, signal, lowcut=0.7, highcut=3.5, order=4):
        """
        Bandpass filter for rPPG signal.
        Default: 0.7-3.5 Hz = 42-210 BPM
        """
        nyquist = self.fps / 2
        low = lowcut / nyquist
        high = highcut / nyquist

        # Clamp to valid range
        low = max(low, 0.01)
        high = min(high, 0.99)

        if low >= high:
            return signal

        b, a = butter(order, [low, high], btype="band")
        filtered = filtfilt(b, a, signal)
        return filtered

    # ---- Heart Rate (HR) ----
    def compute_heart_rate(self, signal):
        """Compute heart rate from rPPG signal using FFT."""
        if len(signal) < self.fps * 2:
            return 0.0

        # Filter signal
        filtered = self.bandpass_filter(signal, lowcut=0.7, highcut=3.5)

        # FFT
        n = len(filtered)
        yf = np.abs(fft(filtered))
        xf = fftfreq(n, d=1.0 / self.fps)

        # Only positive frequencies in HR range
        positive = xf > 0
        hr_range = (xf >= 0.7) & (xf <= 3.5)
        valid = positive & hr_range

        if not np.any(valid):
            return 0.0

        peak_freq = xf[valid][np.argmax(yf[valid])]
        hr_bpm = peak_freq * 60.0

        return hr_bpm

    # ---- Heart Rate Variability (HRV) ----
    def compute_hrv(self, signal):
        """
        Compute HRV metrics from rPPG signal.
        Returns: dict with SDNN, RMSSD, pNN50
        """
        if len(signal) < self.fps * 5:
            return {"SDNN": 0, "RMSSD": 0, "pNN50": 0}

        # Filter signal
        filtered = self.bandpass_filter(signal, lowcut=0.7, highcut=3.5)

        # Normalize
        filtered = (filtered - filtered.mean()) / (filtered.std() + 1e-8)

        # Find peaks (heartbeats)
        min_distance = int(self.fps * 0.4)  # Min 0.4s between beats (~150 BPM)
        peaks, _ = find_peaks(filtered, distance=min_distance, height=0.1)

        if len(peaks) < 3:
            return {"SDNN": 0, "RMSSD": 0, "pNN50": 0}

        # RR intervals in milliseconds
        rr_intervals = np.diff(peaks) / self.fps * 1000

        # Remove outliers (RR < 300ms or > 1500ms)
        rr_intervals = rr_intervals[(rr_intervals > 300) & (rr_intervals < 1500)]

        if len(rr_intervals) < 2:
            return {"SDNN": 0, "RMSSD": 0, "pNN50": 0}

        # SDNN - Standard deviation of RR intervals
        sdnn = np.std(rr_intervals)

        # RMSSD - Root mean square of successive differences
        successive_diff = np.diff(rr_intervals)
        rmssd = np.sqrt(np.mean(successive_diff ** 2))

        # pNN50 - Percentage of successive differences > 50ms
        pnn50 = np.sum(np.abs(successive_diff) > 50) / len(successive_diff) * 100

        return {
            "SDNN": round(sdnn, 2),
            "RMSSD": round(rmssd, 2),
            "pNN50": round(pnn50, 2)
        }

    # ---- Respiratory Rate ----
    def compute_respiratory_rate(self, signal):
        """
        Estimate respiratory rate from rPPG signal.
        Respiration: 0.15-0.4 Hz = 9-24 breaths/min
        """
        if len(signal) < self.fps * 5:
            return 0.0

        # Filter for respiratory range
        filtered = self.bandpass_filter(signal, lowcut=0.15, highcut=0.4)

        # FFT
        n = len(filtered)
        yf = np.abs(fft(filtered))
        xf = fftfreq(n, d=1.0 / self.fps)

        # Respiratory frequency range
        valid = (xf >= 0.15) & (xf <= 0.4)

        if not np.any(valid):
            return 0.0

        peak_freq = xf[valid][np.argmax(yf[valid])]
        resp_rate = peak_freq * 60.0  # breaths per minute

        return round(resp_rate, 1)

    # ---- SpO2 Estimation ----
    def compute_spo2(self, rgb_signals):
        """
        Estimate SpO2 from RGB signals.
        Uses ratio of red to blue channel AC/DC components.
        Note: This is an ESTIMATION - not medical grade.

        Args:
            rgb_signals: numpy array of shape (N, 3) - [R, G, B]
        """
        if len(rgb_signals) < self.fps * 3:
            return 0.0

        red = rgb_signals[:, 0].astype(float)
        blue = rgb_signals[:, 2].astype(float)

        # Filter both channels
        red_filtered = self.bandpass_filter(red, lowcut=0.7, highcut=3.5)
        blue_filtered = self.bandpass_filter(blue, lowcut=0.7, highcut=3.5)

        # AC component (standard deviation of filtered signal)
        red_ac = np.std(red_filtered)
        blue_ac = np.std(blue_filtered)

        # DC component (mean of raw signal)
        red_dc = np.mean(red)
        blue_dc = np.mean(blue)

        if red_dc == 0 or blue_dc == 0 or blue_ac == 0:
            return 0.0

        # Ratio of ratios
        ratio = (red_ac / red_dc) / (blue_ac / blue_dc + 1e-8)

        # Empirical calibration (approximate)
        # SpO2 = A - B * ratio (calibration curve)
        spo2 = 110 - 25 * ratio

        # Clamp to realistic range
        spo2 = np.clip(spo2, 70, 100)

        return round(float(spo2), 1)

    # ---- Signal Quality (SNR) ----
    def compute_snr(self, signal):
        """Compute Signal-to-Noise Ratio of rPPG signal."""
        if len(signal) < self.fps * 2:
            return 0.0

        filtered = self.bandpass_filter(signal, lowcut=0.7, highcut=3.5)

        # Power spectral density
        freqs, psd = welch(filtered, fs=self.fps, nperseg=min(256, len(filtered)))

        # Signal power (HR range: 0.7-3.5 Hz)
        hr_mask = (freqs >= 0.7) & (freqs <= 3.5)

        if not np.any(hr_mask):
            return 0.0

        # Find peak frequency
        peak_idx = np.argmax(psd[hr_mask])
        peak_freq = freqs[hr_mask][peak_idx]

        # Signal band (peak ± 0.2 Hz)
        signal_mask = (freqs >= peak_freq - 0.2) & (freqs <= peak_freq + 0.2)
        noise_mask = hr_mask & ~signal_mask

        signal_power = np.sum(psd[signal_mask])
        noise_power = np.sum(psd[noise_mask])

        if noise_power == 0:
            return 0.0

        snr = 10 * np.log10(signal_power / noise_power)
        return round(float(snr), 2)

    # ---- Extract All Vitals ----
    def extract_all_vitals(self, rppg_signal, rgb_signals=None):
        """
        Extract all vitals from rPPG signal.

        Args:
            rppg_signal: 1D numpy array - extracted rPPG signal
            rgb_signals: 2D numpy array (N, 3) - raw RGB means (for SpO2)

        Returns:
            dict with all vital signs
        """
        vitals = {
            "heart_rate_bpm": round(self.compute_heart_rate(rppg_signal), 1),
            "hrv": self.compute_hrv(rppg_signal),
            "respiratory_rate_bpm": self.compute_respiratory_rate(rppg_signal),
            "signal_quality_snr_db": self.compute_snr(rppg_signal),
        }

        if rgb_signals is not None:
            vitals["spo2_percent"] = self.compute_spo2(rgb_signals)

        return vitals