import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Apply custom CSS for Stray theme
st.markdown("""
    <style>
    body {
        background-color: #0c0c0c;
        color: #f8f8f2;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stApp {
        background-color: #0c0c0c;
    }
    h1 {
        font-size: 48px;
        color: #f9a825; /* Stray orange */
        text-shadow: 0 0 10px #f9a825;
    }
    .stFileUploader label {
        color: #00ffff;
        font-weight: bold;
    }
    .stButton>button {
        background-color: #f9a825;
        color: black;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #ffd740;
    }
    </style>
""", unsafe_allow_html=True)

# App UI
st.title("Sound Detection from Uploaded Audio")

uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file is not None:
    sample_rate, data = wavfile.read(uploaded_file)

    # Mono conversion if stereo
    if len(data.shape) == 2:
        data = data.mean(axis=1)
    data = data / np.max(np.abs(data))  # Normalize

    # Plot waveform
    st.subheader("Waveform")
    time = np.linspace(0, len(data) / sample_rate, num=len(data))
    fig, ax = plt.subplots()
    ax.plot(time, data, color="#00ffff")  # Neon cyan
    ax.set_facecolor("#1e1e1e")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

    # Loudest point
    max_idx = np.argmax(np.abs(data))
    max_time = max_idx / sample_rate
    st.success(f"Loudest sound detected at {max_time:.2f} seconds")

    st.info("This simulation uses one mic. For direction, record from different angles or rotate mic.")
