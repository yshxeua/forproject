import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import correlate
from math import asin, degrees

# Constants
MIC_DISTANCE = 0.2  # Distance between microphones in meters
SPEED_OF_SOUND = 343  # Speed of sound in m/s

st.title("🎤 Direction-of-Arrival Estimation Using Microphone Array")
st.markdown("""
Upload a **stereo WAV** file recorded with two microphones placed apart. This app will:
- Calculate Time Difference of Arrival (TDOA)
- Estimate the angle of sound arrival
- Visualize it on a polar plot
""")

# File uploader
uploaded_file = st.file_uploader("Upload stereo WAV file", type=["wav"])

if uploaded_file:
    # Read audio file
    sample_rate, data = wavfile.read(uploaded_file)

    # Ensure it's stereo
    if len(data.shape) == 2 and data.shape[1] == 2:
        left = data[:, 0]
        right = data[:, 1]

        # Normalize
        left = left / np.max(np.abs(left))
        right = right / np.max(np.abs(right))

        # Cross-correlation
        corr = correlate(left, right, mode='full')
        lags = np.arange(-len(left) + 1, len(left))
        lag = lags[np.argmax(corr)]
        tdoa = lag / sample_rate

        st.write(f"🕒 **TDOA**: {tdoa * 1e6:.2f} microseconds")

        # Estimate angle
        try:
            angle_rad = asin(tdoa * SPEED_OF_SOUND / MIC_DISTANCE)
            angle_deg = degrees(angle_rad)
            st.success(f"📐 Estimated DoA: **{angle_deg:.2f}°**")
        except ValueError:
            st.error("🚫 TDOA out of range. Sound source may be beyond ±90°.")
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
    else:
        st.error("⚠️ Please upload a **stereo** WAV file with two channels.")
