import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

st.title("Sound Detection from Uploaded Audio")

# File uploader
uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file is not None:
    # Read WAV file
    sample_rate, data = wavfile.read(uploaded_file)

    # Convert stereo to mono if necessary
    if len(data.shape) == 2:
        data = data.mean(axis=1)

    # Normalize
    data = data / np.max(np.abs(data))

    # Plot waveform
    st.subheader("Waveform")
    time = np.linspace(0, len(data) / sample_rate, num=len(data))
    fig, ax = plt.subplots()
    ax.plot(time, data)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

    # Detect loudest sound
    max_idx = np.argmax(np.abs(data))
    max_time = max_idx / sample_rate
    st.success(f"Loudest sound detected at {max_time:.2f} seconds")

    # Simulated direction note
    st.info("This simulation assumes one microphone. To guess direction, use multiple recordings from different mic positions.")
