import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# ... [Your existing CSS and style here unchanged] ...

st.title("🔊 DIRECTION OF ARRIVAL ESTIMATION USING MICROPHONE ARRAY")

uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file is not None:
    sample_rate, data = wavfile.read(uploaded_file)

    # Check if stereo or mono
    if len(data.shape) == 2 and data.shape[1] == 2:
        left_channel = data[:, 0]
        right_channel = data[:, 1]

        # Normalize channels
        left_norm = left_channel / np.max(np.abs(left_channel))
        right_norm = right_channel / np.max(np.abs(right_channel))

        # Compare average amplitude
        left_avg = np.mean(np.abs(left_norm))
        right_avg = np.mean(np.abs(right_norm))

        if left_avg > right_avg * 1.1:  # small threshold to avoid ties
            direction_msg = "🔊 Sound is dominant on the **LEFT** channel."
        elif right_avg > left_avg * 1.1:
            direction_msg = "🔊 Sound is dominant on the **RIGHT** channel."
        else:
            direction_msg = "🔊 Sound levels are balanced between LEFT and RIGHT channels."

        # For plotting, average to mono
        data_mono = (left_channel + right_channel) / 2
        data_norm = data_mono / np.max(np.abs(data_mono))

    else:
        direction_msg = "⚠️ Single channel audio detected, left/right direction estimation unavailable."
        # Mono or multi-channel but not stereo: normalize mono
        if len(data.shape) == 2:
            data_mono = data.mean(axis=1)
        else:
            data_mono = data
        data_norm = data_mono / np.max(np.abs(data_mono))

    # Waveform plot
    st.subheader("📈 Waveform")
    time = np.linspace(0, len(data_norm) / sample_rate, num=len(data_norm))
    fig, ax = plt.subplots()
    ax.plot(time, data_norm, color="#00ffff")
    ax.set_facecolor("black")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

    # Loudest peak
    max_idx = np.argmax(np.abs(data_norm))
    max_time = max_idx / sample_rate
    st.success(f"🟣 Loudest point at {max_time:.2f} seconds")

    # Show left/right direction message
    st.info(direction_msg)

    st.info("This is a single-mic analysis. To estimate direction, record from multiple positions.")
