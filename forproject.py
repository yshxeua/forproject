import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import correlate
from math import asin, degrees

# Constants
MIC_DISTANCE = 0.2  # meters
SPEED_OF_SOUND = 343  # m/s

st.title("🎤 Direction-of-Arrival Estimation (Two Mono WAVs)")
st.markdown("""
Upload **two mono WAV files**, each recorded from a separate microphone placed apart. This app will:
- Calculate the Time Difference of Arrival (TDOA)
- Estimate the angle of arrival of sound
- Display the result and polar plot
""")

# Upload two mono WAV files
file1 = st.file_uploader("Upload WAV from Mic 1 (Left)", type=["wav"])
file2 = st.file_uploader("Upload WAV from Mic 2 (Right)", type=["wav"])

if file1 and file2:
    sr1, data1 = wavfile.read(file1)
    sr2, data2 = wavfile.read(file2)

    if sr1 != sr2:
        st.error("⚠️ Sampling rates of both files must match.")
    elif len(data1.shape) != 1 or len(data2.shape) != 1:
        st.error("⚠️ Both files must be mono (single-channel).")
    elif len(data1) != len(data2):
        st.warning("⚠️ Files are different lengths — trimming to shortest.")
        min_len = min(len(data1), len(data2))
        data1 = data1[:min_len]
        data2 = data2[:min_len]
        sample_rate = sr1
    else:
        sample_rate = sr1

    # Normalize signals
    data1 = data1 / np.max(np.abs(data1))
    data2 = data2 / np.max(np.abs(data2))

    # Cross-correlation
    corr = correlate(data1, data2, mode='full')
    lags = np.arange(-len(data1) + 1, len(data1))
    lag = lags[np.argmax(corr)]
    tdoa = lag / sample_rate

    st.write(f"🕒 **TDOA**: {tdoa * 1e6:.2f} microseconds")

    # Angle estimation
    try:
        angle_rad = asin(tdoa * SPEED_OF_SOUND / MIC_DISTANCE)
        angle_deg = degrees(angle_rad)
        st.success(f"📐 Estimated Direction of Arrival: **{angle_deg:.2f}°**")
    except ValueError:
        st.error("🚫 TDOA value too large. Sound came from beyond ±90°.")
        angle_deg = None

    # Polar plot
    if angle_deg is not None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, polar=True)
        ax.set_theta_zero_location('front')
        ax.set_theta_direction(-1)
        ax.plot([0, np.deg2rad(angle_deg)], [0, 1], color='magenta', linewidth=3)
        ax.set_yticklabels([])
        ax.set_title("Estimated Sound Direction", color='blue')
        st.pyplot(fig)
