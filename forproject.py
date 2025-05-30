import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import correlate
from math import asin, degrees

# Constants
MIC_DISTANCE = 0.2  # meters (adjust based on your setup)
SPEED_OF_SOUND = 343  # m/s

# Styling
st.markdown("""
    <style>
    body, .stApp {
        background-image: url("https://raw.githubusercontent.com/yshxeua/forproject/main/1162247.jpg");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: #f0f0f0;
        font-family: 'Orbitron', sans-serif;
    }

    .css-1v3fvcr {
        background-color: rgba(10, 10, 15, 0.75) !important;
        border-radius: 15px;
        padding: 25px;
        box-shadow:
            0 0 10px #ff00ff,
            0 0 20px #ff00ff,
            0 0 30px #00ffff,
            0 0 40px #00ffff;
        border: 2px solid #ff00ff;
        max-width: 800px;
        margin: auto;
    }

    h1 {
        color: #ff00ff;
        text-align: center;
        font-size: 2.5rem;
        text-shadow:
            0 0 10px #ff00ff,
            0 0 20px #ff00ff,
            0 0 30px #ff00ff;
    }

    .stSubheader {
        color: #00ffff;
        text-shadow:
            0 0 6px #00ffff,
            0 0 12px #00ffff;
    }

    .stButton>button {
        background: linear-gradient(90deg, #00ffff, #ff00ff);
        color: #000;
        font-weight: 700;
        font-size: 18px;
        padding: 12px 25px;
        border-radius: 15px;
        border: none;
        box-shadow:
            0 0 10px #00ffff,
            0 0 20px #ff00ff;
        cursor: pointer;
    }

    </style>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# Title
st.title("🔊 DIRECTION OF ARRIVAL ESTIMATION USING MICROPHONE ARRAY")

# File uploader
uploaded_file = st.file_uploader("Upload a stereo WAV file", type=["wav"])

# Processing after file upload
if uploaded_file is not None:
    sample_rate, data = wavfile.read(uploaded_file)

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
        tdoa = lag / sample_rate  # time difference in seconds

        # Direction angle
        try:
            angle_rad = asin(tdoa * SPEED_OF_SOUND / MIC_DISTANCE)
            angle_deg = degrees(angle_rad)
            direction_msg = f"📐 Estimated Direction of Arrival: **{angle_deg:.2f}°**"
        except ValueError:
            direction_msg = "🚫 Sound arrival angle out of detectable range (too large TDOA)."
            angle_deg = None
    else:
        direction_msg = "⚠️ Please upload a **stereo** WAV file recorded with two microphones."
        angle_deg = None

    # Plot waveform
    st.subheader("📈 Waveform")
    mono = data.mean(axis=1) if len(data.shape) == 2 else data
    mono = mono / np.max(np.abs(mono))
    time = np.linspace(0, len(mono) / sample_rate, len(mono))
    fig1, ax1 = plt.subplots()
    ax1.plot(time, mono, color="#00ffff")
    ax1.set_facecolor("black")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    st.pyplot(fig1)

    # Loudest peak
    max_idx = np.argmax(np.abs(mono))
    max_time = max_idx / sample_rate
    st.success(f"🟣 Loudest point at {max_time:.2f} seconds")

    # Direction result
    st.info(direction_msg)

    # Polar plot
    if angle_deg is not None:
        st.subheader("🧭 Polar Plot of Sound Direction")
        fig2 = plt.figure(figsize=(4, 4))
        ax2 = fig2.add_subplot(111, polar=True)
        angle_rad = np.deg2rad(angle_deg)
        ax2.plot([0, angle_rad], [0, 1], color='#ff00ff', linewidth=3)
        ax2.set_yticklabels([])
        ax2.set_theta_zero_location('front')
        ax2.set_theta_direction(-1)
        st.pyplot(fig2)

    st.info("💡 Make sure microphones are well-calibrated and spaced evenly for accurate estimation.")
