import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

st.markdown("""
    <style>
    /* Set background image on the body */
    body {
        background-image: url("https://raw.githubusercontent.com/yshxeua/forproject/main/cyberpunk-bg.jpg");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
        color: #f1f1f1;
    }

    /* Optional: add a transparent overlay to darken background */
    .css-1v3fvcr {
        background-color: rgba(0,0,0,0.6) !important;
        border-radius: 10px;
        padding: 20px;
    }

    h1, .css-1v3fvcr h1 {
        font-family: 'Orbitron', sans-serif;
        font-size: 48px;
        color: #ff00ff;
        text-shadow: 0 0 10px #ff00ff;
        text-align: center;
    }

    /* Customize uploader label */
    .stFileUploader label {
        font-size: 18px;
        color: #00ffff;
        font-weight: bold;
        text-shadow: 0 0 8px #00ffff;
    }

    /* Customize button */
    .stButton>button {
        background-color: #00ffff;
        color: #000;
        border: none;
        border-radius: 10px;
        font-size: 18px;
        padding: 10px 20px;
        box-shadow: 0 0 10px #00ffff;
        transition: background-color 0.3s ease;
    }

    .stButton>button:hover {
        background-color: #ff00ff;
        color: white;
    }

    /* Plot area background */
    .css-1v3fvcr, .css-ffhzg2 {
        background-color: rgba(0,0,0,0.6) !important;
        padding: 20px;
        border-radius: 10px;
    }
    </style>

    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@500&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

st.title("🔊 Cyberpunk Audio Analyzer")

uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file is not None:
    sample_rate, data = wavfile.read(uploaded_file)

    # Mono check
    if len(data.shape) == 2:
        data = data.mean(axis=1)
    data = data / np.max(np.abs(data))

    # Waveform
    st.subheader("📈 Waveform")
    time = np.linspace(0, len(data) / sample_rate, num=len(data))
    fig, ax = plt.subplots()
    ax.plot(time, data, color="#00ffff")
    ax.set_facecolor("black")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

    # Loudest peak
    max_idx = np.argmax(np.abs(data))
    max_time = max_idx / sample_rate
    st.success(f"🟣 Loudest point at {max_time:.2f} seconds")

    st.info("This is a single-mic analysis. To estimate direction, record from multiple positions.")
