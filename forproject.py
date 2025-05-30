import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Load external CSS for faster rendering
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Title
st.title("🔊 DIRECTION OF ARRIVAL ESTIMATION USING MICROPHONE ARRAY")

# File uploader
uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

@st.cache_data
def read_audio(uploaded_file):
    return wavfile.read(uploaded_file)

if uploaded_file is not None:
    status = st.empty()
    status.info("⏳ Processing audio...")

    sample_rate, data = read_audio(uploaded_file)

    # Stereo detection
    if len(data.shape) == 2 and data.shape[1] == 2:
        left_channel = data[:, 0]
        right_channel = data[:, 1]

        # Normalize
        peak_l = np.max(np.abs(left_channel))
        peak_r = np.max(np.abs(right_channel))
        left_norm = left_channel / peak_l if peak_l else left_channel
        right_norm = right_channel / peak_r if peak_r else right_channel

        left_avg = np.mean(np.abs(left_norm))
        right_avg = np.mean(np.abs(right_norm))

        if left_avg > right_avg * 1.1:
            direction_msg = "🔊 Sound is dominant on the **LEFT** channel."
        elif right_avg > left_avg * 1.1:
            direction_msg = "🔊 Sound is dominant on the **RIGHT** channel."
        else:
            direction_msg = "🔊 Sound levels are balanced between LEFT and RIGHT channels."

        data_mono = (left_channel + right_channel) / 2
    else:
        data_mono = data if len(data.shape) == 1 else data.mean(axis=1)
        peak = np.max(np.abs(data_mono))
        data_norm = data_mono / peak if peak else data_mono

        half = len(data_norm) // 2
        left_avg = np.mean(np.abs(data_norm[:half]))
        right_avg = np.mean(np.abs(data_norm[half:]))

        if left_avg > right_avg * 1.1:
            direction_msg = "⚠️ Mono audio — guess: Sound is dominant on the LEFT."
        elif right_avg > left_avg * 1.1:
            direction_msg = "⚠️ Mono audio — guess: Sound is dominant on the RIGHT."
        else:
            direction_msg = "⚠️ Mono audio — Sound levels appear balanced."

    # Normalize for plotting
    peak_mono = np.max(np.abs(data_mono))
    data_norm = data_mono / peak_mono if peak_mono else data_mono

    # Optional waveform
    if st.checkbox("📊 Show waveform plot"):
        st.subheader("📈 Waveform")
        time = np.linspace(0, len(data_norm) / sample_rate, num=len(data_norm))
        fig, ax = plt.subplots(figsize=(10, 3), dpi=100)
        ax.plot(time, data_norm, color="#00ffff")
        ax.set_facecolor("black")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        fig.tight_layout()
        st.pyplot(fig)

    max_idx = np.argmax(np.abs(data_norm))
    max_time = max_idx / sample_rate
    st.success(f"🟣 Loudest point at {max_time:.2f} seconds")
    st.info(direction_msg)

    status.success("✅ Audio processed.")

