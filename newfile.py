import streamlit as st
import numpy as np
import sounddevice as sd
from scipy.signal import correlate
import matplotlib.pyplot as plt

# Constants
SOUND_SPEED = 343  # m/s
MIC_DISTANCE = 0.2  # distance between microphones in meters
DURATION = 1  # seconds
SAMPLERATE = 44100  # Hz

st.title("🔊 Direction-of-Arrival Estimation")
st.write("Using 2 microphones to estimate the angle of arrival of a sound source.")

if st.button("🎙️ Record Sound"):
    st.info("Recording from stereo microphone for 1 second...")
    
    # Record from stereo input (2 channels)
    recording = sd.rec(int(DURATION * SAMPLERATE), samplerate=SAMPLERATE, channels=2)
    sd.wait()
    
    # Separate channels
    mic1 = recording[:, 0]
    mic2 = recording[:, 1]

    # Cross-correlation to estimate TDOA
    correlation = correlate(mic1, mic2, mode='full')
    lag = np.argmax(correlation) - len(mic1) + 1
    tdoa = lag / SAMPLERATE
    
    # Calculate angle
    try:
        sin_theta = (tdoa * SOUND_SPEED) / MIC_DISTANCE
        sin_theta = np.clip(sin_theta, -1.0, 1.0)  # handle edge cases
        angle = np.degrees(np.arcsin(sin_theta))
    except ValueError:
        angle = None

    # Display results
    st.write(f"Estimated Time Difference of Arrival (TDOA): {tdoa:.6f} seconds")
    if angle is not None:
        st.success(f"Estimated Angle of Arrival: {angle:.2f}°")
    else:
        st.error("Could not calculate angle due to invalid TDOA.")

    # Polar plot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    if angle is not None:
        angle_rad = np.radians(angle)
        ax.plot([angle_rad, angle_rad], [0, 1], linewidth=3)
    ax.set_title("Estimated Direction")
    st.pyplot(fig)
