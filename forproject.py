import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import av
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from math import asin, degrees

# Constants
MIC_DISTANCE = 0.2  # distance between mics in meters
SPEED_OF_SOUND = 343  # speed of sound in m/s
SAMPLE_RATE = 48000  # WebRTC audio sample rate

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

class AudioProcessor(AudioProcessorBase):
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray()
        
        # Check stereo input
        if audio.shape[0] < 2:
            return frame  # Mono or not enough data
        
        # Separate channels
        left = audio[0::2]
        right = audio[1::2]

        # Normalize
        if np.max(np.abs(left)) > 0:
            left = left / np.max(np.abs(left))
        if np.max(np.abs(right)) > 0:
            right = right / np.max(np.abs(right))

        # Cross-correlation
        corr = correlate(left, right, mode='full')
        lags = np.arange(-len(left) + 1, len(left))
        lag = lags[np.argmax(corr)]
        tdoa = lag / SAMPLE_RATE

        # Angle estimation
        try:
            angle_rad = asin(tdoa * SPEED_OF_SOUND / MIC_DISTANCE)
            angle_deg = degrees(angle_rad)
        except ValueError:
            angle_deg = None

        # Plot waveforms
        t = np.linspace(0, len(left) / SAMPLE_RATE, len(left))
        fig, axs = plt.subplots(2, 1, figsize=(6, 3), sharex=True)
        axs[0].plot(t, left, color='blue')
        axs[0].set_title("Mic 1 (Left)")
        axs[1].plot(t, right, color='red')
        axs[1].set_title("Mic 2 (Right)")
        axs[1].set_xlabel("Time [s]")
        for ax in axs:
            ax.grid(True)
        placeholder_waveform.pyplot(fig)

        # Show TDOA
        placeholder_tdoa.markdown(f"🕒 **TDOA**: `{tdoa * 1e6:.2f}` microseconds")

        # Show angle
        if angle_deg is not None and abs(tdoa * SPEED_OF_SOUND) <= MIC_DISTANCE:
            placeholder_angle.success(f"📐 Estimated Angle: `{angle_deg:.2f}°`")
            
            # Polar plot
            fig2 = plt.figure(figsize=(4, 4))
            ax = fig2.add_subplot(111, polar=True)
            ax.set_theta_zero_location('front')
            ax.set_theta_direction(-1)
            ax.plot([0, np.deg2rad(angle_deg)], [0, 1], color='magenta', linewidth=3)
            ax.set_yticklabels([])
            ax.set_title("Direction of Arrival")
            placeholder_polar.pyplot(fig2)
        else:
            placeholder_angle.error("🚫 Angle estimation failed. Use stereo input.")
            placeholder_polar.empty()

        return frame

# Start the WebRTC mic stream
webrtc_streamer(
    key="doa-audio",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False}
)
