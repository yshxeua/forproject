import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import av
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from math import asin, degrees
import time

# Constants
MIC_DISTANCE = 0.2  # distance between mics in meters
SPEED_OF_SOUND = 343  # m/s
SAMPLE_RATE = 48000  # Hz
SOUND_THRESHOLD = 0.001  # lowered for sensitivity

# Page setup
st.set_page_config(page_title="Real-Time DoA Estimation", layout="centered")
st.title("🎤 Real-Time Direction-of-Arrival Estimation")

# UI placeholders
placeholder_waveform = st.empty()
placeholder_tdoa = st.empty()
placeholder_angle = st.empty()
placeholder_polar = st.empty()
placeholder_status = st.empty()

# Helper to check if sound is significant
def is_sound_present(signal, threshold=SOUND_THRESHOLD):
    rms = np.sqrt(np.mean(signal**2))
    return rms > threshold

# Audio Processor
class AudioProcessor(AudioProcessorBase):
    latest_audio = None
    latest_tdoa = None
    latest_angle = None
    sound_detected = False
    input_channels = 0

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray()

        # Debugging info
        print(f"Audio shape: {audio.shape}")
        print(f"RMS: {np.sqrt(np.mean(audio**2))}")

        if len(audio.shape) == 1:
            # Mono input
            self.input_channels = 1
            mono = audio
            if np.max(np.abs(mono)) > 0:
                mono = mono / np.max(np.abs(mono))
            self.sound_detected = is_sound_present(mono)
            AudioProcessor.latest_audio = audio.copy() if self.sound_detected else None
            AudioProcessor.latest_tdoa = None
            AudioProcessor.latest_angle = 0.0

        elif len(audio.shape) == 2 and audio.shape[0] >= 2:
            # Stereo input
            self.input_channels = audio.shape[0]
            left = audio[0]
            right = audio[1]
            left = left / np.max(np.abs(left)) if np.max(np.abs(left)) > 0 else left
            right = right / np.max(np.abs(right)) if np.max(np.abs(right)) > 0 else right

            self.sound_detected = is_sound_present(left) or is_sound_present(right)
            AudioProcessor.latest_audio = audio.copy() if self.sound_detected else None

            if self.sound_detected:
                corr = correlate(left, right, mode='full')
                lags = np.arange(-len(left) + 1, len(left))
                lag = lags[np.argmax(corr)]
                tdoa = lag / SAMPLE_RATE
                AudioProcessor.latest_tdoa = tdoa

                try:
                    angle_rad = asin(tdoa * SPEED_OF_SOUND / MIC_DISTANCE)
                    angle_deg = degrees(angle_rad)
                except ValueError:
                    angle_deg = None
                AudioProcessor.latest_angle = angle_deg
            else:
                AudioProcessor.latest_tdoa = None
                AudioProcessor.latest_angle = None

        else:
            # Invalid or unsupported input
            self.sound_detected = False
            AudioProcessor.latest_audio = None
            AudioProcessor.latest_tdoa = None
            AudioProcessor.latest_angle = None

        return frame

# Plotting waveform
def plot_waveform(audio):
    if audio is None:
        placeholder_waveform.empty()
        return

    if len(audio.shape) == 1:
        mono = audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio
        t = np.linspace(0, len(mono) / SAMPLE_RATE, len(mono))
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.plot(t, mono, color='blue')
        ax.set_title("Mono Mic Input")
        ax.set_xlabel("Time [s]")
        ax.grid(True)
        placeholder_waveform.pyplot(fig)

    elif len(audio.shape) == 2 and audio.shape[0] >= 2:
        left = audio[0]
        right = audio[1]
        left = left / np.max(np.abs(left)) if np.max(np.abs(left)) > 0 else left
        right = right / np.max(np.abs(right)) if np.max(np.abs(right)) > 0 else right
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

# Plot TDOA value
def plot_tdoa(tdoa):
    if tdoa is not None:
        placeholder_tdoa.markdown(f"🕒 **TDOA**: `{tdoa * 1e6:.2f}` microseconds")
    else:
        placeholder_tdoa.empty()

# Plot angle and polar chart
def plot_angle(angle_deg, is_mono=False):
    if is_mono:
        placeholder_angle.info("ℹ️ Mono input detected — DoA estimation shown as fixed 0°.")
        angle_deg = 0.0

    if angle_deg is None:
        placeholder_angle.info("ℹ️ No angle estimated yet.")
        placeholder_polar.empty()
        return

    fig2 = plt.figure(figsize=(4, 4))
    ax = fig2.add_subplot(111, polar=True)
    ax.set_theta_zero_location('front')
    ax.set_theta_direction(-1)
    ax.plot([0, np.deg2rad(angle_deg)], [0, 1], color='magenta', linewidth=3)
    ax.set_yticklabels([])
    ax.set_title("Direction of Arrival")
    placeholder_polar.pyplot(fig2)

# WebRTC with STUN configuration
webrtc_ctx = webrtc_streamer(
    key="doa-audio",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)

# UI loop
import time
while True:
    if AudioProcessor.latest_audio is not None:
        plot_waveform(AudioProcessor.latest_audio)

    if AudioProcessor.input_channels == 1:
        plot_angle(None, is_mono=True)
        placeholder_tdoa.empty()
    else:
        plot_tdoa(AudioProcessor.latest_tdoa)
        plot_angle(AudioProcessor.latest_angle)

    if hasattr(webrtc_ctx.audio_processor, "sound_detected"):
        if webrtc_ctx.audio_processor.sound_detected:
            placeholder_status.success("🔊 Sound detected")
        else:
            placeholder_status.info("🔇 No significant sound detected")

    time.sleep(1)
