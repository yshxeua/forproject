import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import av
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from math import asin, degrees

# Constants
MIC_DISTANCE = 0.2  # meters
SPEED_OF_SOUND = 343
SAMPLE_RATE = 48000

st.set_page_config(page_title="Real-Time DoA Estimation", layout="centered")
st.title("🎤 Real-Time Direction-of-Arrival Estimation")

st.markdown("""
This app captures audio **in real time** using a stereo mic setup (like AirPods or stereo USB mics),  
then estimates the **direction of arrival (DoA)** of the sound using **TDOA** and **cross-correlation**.
""")

# Placeholders for dynamic content
placeholder_waveform = st.empty()
placeholder_tdoa = st.empty()
placeholder_angle = st.empty()
placeholder_polar = st.empty()

# Shared state to hold latest audio snippet
class AudioProcessor(AudioProcessorBase):
    latest_audio = None  # store latest audio frame array
    
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray()
        
        if audio.shape[0] < 2:
            return frame
        
        # Save latest stereo audio snippet (copy so no overwrite)
        AudioProcessor.latest_audio = audio.copy()
        
        # Continue your existing processing for DoA (optional here)
        return frame

def plot_waveform(audio):
    if audio is None:
        st.warning("No audio captured yet.")
        return

    left = audio[0::2]
    right = audio[1::2]

    # Normalize
    if np.max(np.abs(left)) > 0:
        left = left / np.max(np.abs(left))
    if np.max(np.abs(right)) > 0:
        right = right / np.max(np.abs(right))

    t = np.linspace(0, len(left) / SAMPLE_RATE, len(left))
    fig, axs = plt.subplots(2, 1, figsize=(6, 3), sharex=True)
    axs[0].plot(t, left, color='blue')
    axs[0].set_title("Mic 1 (Left)")
    axs[1].plot(t, right, color='red')
    axs[1].set_title("Mic 2 (Right)")
    axs[1].set_xlabel("Time [s]")
    for ax in axs:
        ax.grid(True)
    st.pyplot(fig)

webrtc_ctx = webrtc_streamer(
    key="doa-audio",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False}
)

if st.button("Show Mic Waveform"):
    if AudioProcessor.latest_audio is not None:
        plot_waveform(AudioProcessor.latest_audio)
    else:
        st.warning("Audio not captured yet, please speak or make some sound.")
