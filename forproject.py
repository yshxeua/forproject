import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Cyberpunk style CSS with updated background image
st.markdown("""
    <style>
    /* Background */
    body, .stApp {
        background-image: url("https://raw.githubusercontent.com/yshxeua/forproject/main/1162247.jpg");
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

    if len(data.shape) == 2:
        left = data[:, 0]
        right = data[:, 1]

        left_norm = left / np.max(np.abs(left))
        right_norm = right / np.max(np.abs(right))

        diff = np.mean(np.abs(left_norm) - np.abs(right_norm))

        if diff > 0.05:
            direction = "Left"
        elif diff < -0.05:
            direction = "Right"
        else:
            direction = "Center/Indistinct"

        data = data.mean(axis=1)  # convert to mono for plotting

        # Show the direction clearly on the page:
        st.success(f"🟢 Stereo audio detected. Estimated sound direction: **{direction}**")

    else:
        st.warning("⚠️ Single channel audio detected, left/right direction estimation unavailable.")
        data = data / np.max(np.abs(data))
        direction = None

    # Normalize audio for plotting
    data = data / np.max(np.abs(data))

    # Waveform plot
    st.subheader("📈 Waveform")
    time = np.linspace(0, len(data) / sample_rate, num=len(data))
    fig, ax = plt.subplots()
    ax.plot(time, data, color="#00ffff")
    ax.set_facecolor("black")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

    # Loudest peak detection
    max_idx = np.argmax(np.abs(data))
    max_time = max_idx / sample_rate
    st.success(f"🟣 Loudest point at {max_time:.2f} seconds")

    st.info("This is a single-mic analysis. To estimate direction more precisely, record from multiple microphones or channels.")
