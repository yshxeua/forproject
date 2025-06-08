import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import correlate
from math import asin, degrees

# Constants
MIC_DISTANCE = 0.2  # meters
SPEED_OF_SOUND = 343  # m/s

st.set_page_config(page_title="DoA Estimation", layout="centered")
st.title("🎤 Direction-of-Arrival Estimation with Auto Mono-to-Stereo Conversion")

st.markdown("""
Upload **two WAV files** (mono or stereo).  
- If mono, each will be converted to stereo.  
- Left channel from file 1 and right channel from file 2 will be used.  
The app displays:
- Signal waveforms
- Time Difference of Arrival (TDOA)
- Estimated angle of arrival
- Polar plot visualization
""")

# --- Helper: Convert mono to stereo
def ensure_stereo(data):
    if data.ndim == 1:
        return np.stack((data, data), axis=-1)
    elif data.shape[1] == 1:
        return np.repeat(data, 2, axis=1)
    return data

# --- Upload files
file1 = st.file_uploader("Upload WAV file 1", type=["wav"])
file2 = st.file_uploader("Upload WAV file 2", type=["wav"])

if file1 and file2:
    try:
        sr1, data1 = wavfile.read(file1)
        sr2, data2 = wavfile.read(file2)
    except Exception as e:
        st.error(f"❌ Error reading files: {e}")
    else:
        if sr1 != sr2:
            st.error("⚠️ Sampling rates do not match.")
        else:
            data1 = ensure_stereo(data1)
            data2 = ensure_stereo(data2)

            # Extract relevant channels
            signal1 = data1[:, 0]  # Left from file1
            signal2 = data2[:, 1]  # Right from file2

            min_len = min(len(signal1), len(signal2))
            signal1 = signal1[:min_len]
            signal2 = signal2[:min_len]

            # Normalize
            if np.max(np.abs(signal1)) > 0:
                signal1 = signal1 / np.max(np.abs(signal1))
            if np.max(np.abs(signal2)) > 0:
                signal2 = signal2 / np.max(np.abs(signal2))

            # --- Plot waveforms
            time = np.linspace(0, min_len / sr1, min_len)
            fig, axs = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
            axs[0].plot(time, signal1, color='cyan')
            axs[0].set_title("Left Channel (File 1)")
            axs[0].set_ylabel("Amplitude")
            axs[0].grid(True)
            axs[1].plot(time, signal2, color='magenta')
            axs[1].set_title("Right Channel (File 2)")
            axs[1].set_ylabel("Amplitude")
            axs[1].set_xlabel("Time (s)")
            axs[1].grid(True)
            st.pyplot(fig)

            # --- Cross-correlation
            corr = correlate(signal1, signal2, mode='full')
            lags = np.arange(-len(signal1) + 1, len(signal1))
            lag = lags[np.argmax(corr)]
            tdoa = lag / sr1
            st.write(f"🕒 **TDOA**: `{tdoa * 1e6:.2f}` microseconds")

            # --- Angle Estimation
            try:
                angle_rad = asin(tdoa * SPEED_OF_SOUND / MIC_DISTANCE)
                angle_deg = degrees(angle_rad)
                st.success(f"📐 Estimated Direction of Arrival: **{angle_deg:.2f}°**")
            except ValueError:
                st.error("🚫 TDOA too large — sound source likely beyond ±90°")
                angle_deg = None

            # --- Polar plot
            if angle_deg is not None:
                fig2 = plt.figure(figsize=(5, 5))
                ax = fig2.add_subplot(111, polar=True)
                ax.set_theta_zero_location('front')
                ax.set_theta_direction(-1)
                ax.plot([0, np.deg2rad(angle_deg)], [0, 1], color='lime', linewidth=3)
                ax.set_yticklabels([])
                ax.set_title("Estimated Sound Direction", fontsize=14, color='navy')
                st.pyplot(fig2)
