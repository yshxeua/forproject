import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Cyberpunk style CSS
st.markdown("""
    <style>
    /* Background */
    body, .stApp {
        background-image: url("https://raw.githubusercontent.com/yshxeua/forproject/main/cyberpunk-bg.jpg");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: #f0f0f0;
        font-family: 'Orbitron', sans-serif;
    }

    /* Main container with translucent black and glowing border */
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

    /* Title styling */
    h1, .css-1v3fvcr h1 {
        color: #ff00ff;
        text-align: center;
        font-size: 3rem;
        text-shadow:
          0 0 10px #ff00ff,
          0 0 20px #ff00ff,
          0 0 30px #ff00ff;
        margin-bottom: 1rem;
    }

    /* Subheaders with neon glow */
    h2, h3, .stSubheader {
        color: #00ffff;
        text-shadow:
          0 0 6px #00ffff,
          0 0 12px #00ffff;
        font-weight: 700;
    }

    /* File uploader label */
    .stFileUploader label {
        font-size: 18px;
        color: #00ffff;
        font-weight: 700;
        text-shadow:
          0 0 8px #00ffff;
    }

    /* Buttons neon style */
    .stButton > button {
        background: linear-gradient(90deg, #00ffff, #ff00ff);
        color: #000;
        font-weight: 700;
        font-size: 18px;
        padding: 12px 25px;
        border-radius: 15px;
        border: none;
        box-shadow:
          0 0 10px #00ffff,
          0 0 20px #ff00ff,
          0 0 30px #ff00ff;
        transition: all 0.3s ease;
        cursor: pointer;
    }

    .stButton > button:hover {
        background: linear-gradient(90deg, #ff00ff, #00ffff);
        color: #fff;
        box-shadow:
          0 0 20px #ff00ff,
          0 0 30px #00ffff,
          0 0 40px #00ffff,
          0 0 50px #ff00ff;
        transform: scale(1.05);
    }

    /* Plot area container */
    .css-ffhzg2 {
        background-color: rgba(10, 10, 15, 0.8) !important;
        border-radius: 15px;
        padding: 15px;
        box-shadow:
          0 0 10px #00ffff,
          0 0 20px #00ffff;
        margin-bottom: 20px;
    }

    /* Axes labels styling (matplotlib) */
    .matplotlib.axes-label {
        color: #00ffff !important;
    }

</style>

<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# Title inside container for centering & styling
st.title("🔊 DIRECTION OF ARRIVAL ESTIMATION USING MICROPHONE ARRAY")

uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file is not None:
    sample_rate, data = wavfile.read(uploaded_file)

    if len(data.shape) == 2 and data.shape[1] == 2:
        # Stereo audio: use real left/right channel comparison
        left_channel = data[:, 0]
        right_channel = data[:, 1]

        # Normalize channels
        left_norm = left_channel / np.max(np.abs(left_channel))
        right_norm = right_channel / np.max(np.abs(right_channel))

        # Average absolute amplitude per channel
        left_avg = np.mean(np.abs(left_norm))
        right_avg = np.mean(np.abs(right_norm))

        if left_avg > right_avg * 1.1:
            direction_msg = "🔊 Sound is dominant on the **LEFT** channel."
        elif right_avg > left_avg * 1.1:
            direction_msg = "🔊 Sound is dominant on the **RIGHT** channel."
        else:
            direction_msg = "🔊 Sound levels are balanced between LEFT and RIGHT channels."

        # Mix down to mono for waveform plotting
        data_mono = (left_channel + right_channel) / 2
        data_norm = data_mono / np.max(np.abs(data_mono))

    else:
        # Mono or non-stereo audio: heuristic based on waveform halves
        data_mono = data if len(data.shape) == 1 else data.mean(axis=1)
        data_norm = data_mono / np.max(np.abs(data_mono))

        half = len(data_norm) // 2
        left_avg = np.mean(np.abs(data_norm[:half]))
        right_avg = np.mean(np.abs(data_norm[half:]))

        if left_avg > right_avg * 1.1:
            direction_msg = "⚠️ Mono audio detected — heuristic guess: Sound is dominant on the LEFT."
        elif right_avg > left_avg * 1.1:
            direction_msg = "⚠️ Mono audio detected — heuristic guess: Sound is dominant on the RIGHT."
        else:
            direction_msg = "⚠️ Mono audio detected — heuristic guess: Sound levels appear balanced."

    # Plot waveform
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

    # Display direction message
    st.info(direction_msg)

    st.info("This is a single-mic analysis. To estimate direction, record from multiple positions.")
