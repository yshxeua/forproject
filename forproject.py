import streamlit as st

import numpy as np

import matplotlib.pyplot as plt

from scipy.signal import correlate

from math import asin, degrees

import sounddevice as sd

import time
 
# Constants

MIC_DISTANCE = 0.2  # meters between microphones

SPEED_OF_SOUND = 343  # m/s

SAMPLE_RATE = 44100  # audio sampling rate

DURATION = 1.0  # seconds per sample block
 
st.title("🎤 Real-Time Direction-of-Arrival Estimation")

st.markdown("""

This app captures real-time audio using a stereo microphone setup (like AirPods L/R).  

It calculates **TDOA** via cross-correlation and estimates the **angle of arrival** in degrees.

""")
 
# Control to start/stop

start_stream = st.toggle("🎧 Start Real-Time Estimation")
 
# Streamlit display placeholders

placeholder_waveform = st.empty()

placeholder_tdoa = st.empty()

placeholder_angle = st.empty()

placeholder_polar = st.empty()
 
def estimate_doa(signal1, signal2, sample_rate):

    # Normalize

    signal1 = signal1 / np.max(np.abs(signal1))

    signal2 = signal2 / np.max(np.abs(signal2))
 
    # Cross-correlation

    corr = correlate(signal1, signal2, mode='full')

    lags = np.arange(-len(signal1) + 1, len(signal1))

    lag = lags[np.argmax(corr)]

    tdoa = lag / sample_rate
 
    # Estimate angle

    try:

        angle_rad = asin(tdoa * SPEED_OF_SOUND / MIC_DISTANCE)

        angle_deg = degrees(angle_rad)

    except ValueError:

        angle_deg = None

    return tdoa, angle_deg
 
if start_stream:

    st.info("🎙️ Listening... Make sounds from different angles.")

    while start_stream:

        try:

            audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=2, dtype='float32')

            sd.wait()
 
            left = audio[:, 0]

            right = audio[:, 1]
 
            tdoa, angle = estimate_doa(left, right, SAMPLE_RATE)
 
            # Waveform plot

            fig, axs = plt.subplots(2, 1, figsize=(8, 4), sharex=True)

            t = np.linspace(0, DURATION, len(left))

            axs[0].plot(t, left, color='blue')

            axs[0].set_title("Mic 1 (Left)")

            axs[1].plot(t, right, color='red')

            axs[1].set_title("Mic 2 (Right)")

            axs[1].set_xlabel("Time [s]")

            for ax in axs: ax.grid(True)

            placeholder_waveform.pyplot(fig)
 
            # Display TDOA

            placeholder_tdoa.write(f"🕒 **TDOA**: {tdoa * 1e6:.2f} microseconds")
 
            # Display angle

            if angle is not None and abs(tdoa * SPEED_OF_SOUND) <= MIC_DISTANCE:

                placeholder_angle.success(f"📐 Estimated Angle: **{angle:.2f}°**")

                # Polar plot

                fig2 = plt.figure(figsize=(4, 4))

                ax = fig2.add_subplot(111, polar=True)

                ax.set_theta_zero_location('front')

                ax.set_theta_direction(-1)

                ax.plot([0, np.deg2rad(angle)], [0, 1], color='magenta', linewidth=3)

                ax.set_yticklabels([])

                ax.set_title("Direction of Arrival")

                placeholder_polar.pyplot(fig2)

            else:

                placeholder_angle.error("🚫 Angle estimation failed. Try again with clearer signal.")

                placeholder_polar.empty()
 
            time.sleep(0.1)  # prevent overloading the CPU

        except Exception as e:

            st.error(f"🎤 Error: {e}")

            break

 
