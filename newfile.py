import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import correlate, windows
from math import asin, degrees
import sys

# Constants
MIC_DISTANCE = 0.2  # meters
SPEED_OF_SOUND = 343  # m/s

st.title("🎤 Direction-of-Arrival Estimation (Improved Accuracy)")
st.markdown("""
Upload **two WAV files** (mono or stereo). If mono, they will be converted to stereo by duplicating the channel.
- Left channel from File 1 and right channel from File 2 are used for DoA estimation.
- Windowing, DC removal, sub-sample interpolation, and polar plots are included for better accuracy.
""")

def ensure_stereo(data):
    """Convert mono to stereo by duplicating the channel if needed."""
    if data.ndim == 1:
        return np.stack((data, data), axis=-1)
    elif data.shape[1] == 1:
        return np.repeat(data, 2, axis=1)
    return data

def sub_sample_peak(corr, lags):
    """Estimate the peak with parabolic interpolation for sub-sample accuracy."""
    peak_idx = np.argmax(corr)
    if 1 < peak_idx < len(corr) - 2:
        y0, y1, y2 = corr[peak_idx - 1], corr[peak_idx], corr[peak_idx + 1]
        denom = (y0 - 2 * y1 + y2)
        if denom != 0:
            delta = 0.5 * (y0 - y2) / denom
        else:
            delta = 0
        refined_lag = lags[peak_idx] + delta
        return refined_lag
    return lags[peak_idx]

file1 = st.file_uploader("Upload WAV file 1", type=["wav"])
file2 = st.file_uploader("Upload WAV file 2", type=["wav"])

if file1 and file2:
    sr1, data1 = wavfile.read(file1)
    sr2, data2 = wavfile.read(file2)

    if sr1 != sr2:
        st.error("⚠️ Sampling rates do not match.")
    else:
        data1 = ensure_stereo(data1)
        data2 = ensure_stereo(data2)

        signal1 = data1[:, 0].astype(np.float64)
        signal2 = data2[:, 1].astype(np.float64)

        min_len = min(len(signal1), len(signal2))
        signal1 = signal1[:min_len]
        signal2 = signal2[:min_len]

        # Normalize and zero-mean
        signal1 -= np.mean(signal1)
        signal2 -= np.mean(signal2)
        signal1 /= np.max(np.abs(signal1))
        signal2 /= np.max(np.abs(signal2))

        # Apply Hann window
        window = windows.hann(min_len)
        signal1 *= window
        signal2 *= window

        # Plot signals
        time = np.linspace(0, min_len / sr1, min_len)
        fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axs[0].plot(time, signal1, color='blue')
        axs[0].set_title("Left Channel (File 1)")
        axs[0].set_ylabel("Amplitude")
        axs[0].grid(True)
        axs[1].plot(time, signal2, color='red')
        axs[1].set_title("Right Channel (File 2)")
        axs[1].set_ylabel("Amplitude")
        axs[1].set_xlabel("Time [s]")
        axs[1].grid(True)
        st.pyplot(fig)

        # Cross-correlation with interpolation
        corr = correlate(signal1, signal2, mode='full')
        lags = np.arange(-len(signal1) + 1, len(signal1))
        refined_lag = sub_sample_peak(corr, lags)
        tdoa = refined_lag / sr1
        st.write(f"🕒 **Refined TDOA**: {tdoa * 1e6:.2f} microseconds")

        # Angle estimation
        try:
            angle_rad = asin(tdoa * SPEED_OF_SOUND / MIC_DISTANCE)
            angle_deg = degrees(angle_rad)
            st.success(f"📐 Estimated Direction of Arrival: **{angle_deg:.2f}°**")
        except ValueError:
            st.error("🚫 TDOA too large; sound source likely beyond ±90°.")
            angle_deg = None

        # Polar plot
        if angle_deg is not None:
            fig2 = plt.figure(figsize=(5, 5))
            ax = fig2.add_subplot(111, polar=True)
            ax.set_theta_zero_location('front')
            ax.set_theta_direction(-1)
            ax.plot([0, np.deg2rad(angle_deg)], [0, 1], color='magenta', linewidth=3)
            ax.set_yticklabels([])
            ax.set_title("Estimated Sound Direction", color='blue')
            st.pyplot(fig2)
