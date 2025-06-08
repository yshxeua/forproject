import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import correlate
from math import asin, degrees

# Constants
MIC_DISTANCE = 0.2  # meters
SPEED_OF_SOUND = 343  # m/s

st.title("🎤 Direction-of-Arrival Estimation with Auto Mono-to-Stereo Conversion")

st.markdown("""
Upload **two WAV files** (mono or stereo). If mono, the app will convert them to stereo by duplicating the channel.

- Left channel from **file 1** and right channel from **file 2** are used for TDOA estimation.
- Results include:
    - Signal waveforms
    - Estimated TDOA (in µs)
    - Estimated arrival angle (in degrees)
    - Polar plot visualization
""")

# File uploaders
file1 = st.file_uploader("Upload WAV file 1 (LEFT channel used)", type=["wav"])
file2 = st.file_uploader("Upload WAV file 2 (RIGHT channel used)", type=["wav"])

def ensure_stereo(data):
    """Convert mono signal to stereo by duplicating the channel."""
    if data.ndim == 1:
        return np.stack((data, data), axis=-1)
    elif data.shape[1] == 1:
        return np.repeat(data, 2, axis=1)
    return data

if file1 and file2:
    sr1, data1 = wavfile.read(file1)
    sr2, data2 = wavfile.read(file2)

    if sr1 != sr2:
        st.error("⚠️ Sampling rates of both files must match.")
    else:
        # Ensure both signals are stereo
        data1 = ensure_stereo(data1)
        data2 = ensure_stereo(data2)

        # Extract left from file 1, right from file 2
        left = data1[:, 0]
        right = data2[:, 1]

        # Trim to same length
        min_len = min(len(left), len(right))
        left = left[:min_len]
        right = right[:min_len]

        # Normalize
        left = left / np.max(np.abs(left))
        right = right / np.max(np.abs(right))

        # Cross-correlation
        corr = correlate(left, right, mode='full')
        lags = np.arange(-len(left) + 1, len(left))
        lag = lags[np.argmax(corr)]
        tdoa = lag / sr1

        st.subheader("📊 Cross-Correlation Result")
        st.write(f"🕒 **TDOA**: {tdoa * 1e6:.2f} microseconds")

        # Angle estimation
        try:
            angle_rad = asin(tdoa * SPEED_OF_SOUND / MIC_DISTANCE)
            angle_deg = degrees(angle_rad)
            st.success(f"📐 Estimated Direction of Arrival: **{angle_deg:.2f}°**")
        except ValueError:
            angle_deg = None
            st.error("🚫 TDOA too large; source likely beyond ±90° or incorrect mic spacing.")

        # Plot signals
        st.subheader("📈 Input Signal Waveforms")
        fig, ax = plt.subplots(figsize=(10, 4))
        time = np.linspace(0, min_len / sr1, min_len)
        ax.plot(time, left, label="Left Mic (File 1)", color='cyan')
        ax.plot(time, right, label="Right Mic (File 2)", color='magenta', alpha=0.7)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.legend()
        ax.set_title("Normalized Audio Signals")
        st.pyplot(fig)

        # Polar plot
        if angle_deg is not None:
            st.subheader("🧭 Polar Plot of Estimated Direction")
            fig2 = plt.figure(figsize=(5, 5))
            ax2 = fig2.add_subplot(111, polar=True)
            ax2.set_theta_zero_location('front')
            ax2.set_theta_direction(-1)
            ax2.plot([0, np.deg2rad(angle_deg)], [0, 1], color='magenta', linewidth=3)
            ax2.set_yticklabels([])
            ax2.set_title("Estimated Sound Direction", color='blue')
            st.pyplot(fig2)

        st.info("✅ Ensure both files are recorded with correct mic positions and the same environment.")
