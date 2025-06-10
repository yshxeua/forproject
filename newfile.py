import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import correlate
from math import asin, degrees
from pydub import AudioSegment
import io

# Constants
MIC_DISTANCE = 0.2  # meters
SPEED_OF_SOUND = 343  # m/s

st.title("🎤 Direction-of-Arrival Estimation with Auto Mono-to-Stereo Conversion")
st.markdown("""
Upload **two audio files** (WAV or M4A). If mono, the app will convert them to stereo by duplicating the channel.
- Left channel from file 1 and right channel from file 2 are used for DoA estimation.
- Waveforms, TDOA, estimated angle, and polar plot will be displayed.
""")

def ensure_stereo(data):
    """Convert mono to stereo by duplicating the channel if needed."""
    if data.ndim == 1:
        data = np.stack((data, data), axis=-1)
    elif data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    return data

def load_audio(file):
    """Load WAV or M4A file and return sample_rate, np.ndarray data."""
    if file.name.endswith(".wav"):
        sr, data = wavfile.read(file)
        return sr, data
    elif file.name.endswith(".m4a"):
        audio = AudioSegment.from_file(file, format="m4a")
        sr = audio.frame_rate
        samples = np.array(audio.get_array_of_samples())
        if audio.channels == 2:
            data = samples.reshape((-1, 2))
        else:
            data = samples
        return sr, data
    else:
        raise ValueError("Unsupported file type")

# File upload
file1 = st.file_uploader("Upload Audio File 1", type=["wav", "m4a"])
file2 = st.file_uploader("Upload Audio File 2", type=["wav", "m4a"])

if file1 and file2:
    sr1, data1 = load_audio(file1)
    sr2, data2 = load_audio(file2)

    if sr1 != sr2:
        st.error("⚠️ Sampling rates do not match.")
    else:
        data1 = ensure_stereo(data1)
        data2 = ensure_stereo(data2)

        # Use left channel from file1 and right channel from file2
        signal1 = data1[:, 0]
        signal2 = data2[:, 1]

        min_len = min(len(signal1), len(signal2))
        signal1 = signal1[:min_len]
        signal2 = signal2[:min_len]

        # Normalize
        signal1 = signal1 / np.max(np.abs(signal1))
        signal2 = signal2 / np.max(np.abs(signal2))

        # Plot waveforms
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

        # Cross-correlation
        corr = correlate(signal1, signal2, mode='full')
        lags = np.arange(-len(signal1) + 1, len(signal1))
        lag = lags[np.argmax(corr)]
        tdoa = lag / sr1
        st.write(f"🕒 **TDOA**: {tdoa * 1e6:.2f} microseconds")

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
