import streamlit as st
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt

st.title("Single Microphone Sound Detection")

duration = st.slider("Recording duration (seconds)", 1, 10, 3)

if st.button("Start Recording"):
    st.write("Recording...")
    fs = 44100  # Sampling frequency
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    st.write("Recording complete.")

    # Flatten the array
    signal = recording.flatten()

    # Plot the waveform
    st.subheader("Waveform")
    fig, ax = plt.subplots()
    time = np.linspace(0, duration, len(signal))
    ax.plot(time, signal)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

    # Find loudest moment
    max_idx = np.argmax(np.abs(signal))
    max_time = max_idx / fs
    st.success(f"Loudest sound detected at {max_time:.2f} seconds")

    # Simulated direction output
    st.info("⚠ With only one mic, we can't detect direction directly.")
    st.write("You could rotate the mic and log loudest response to guess direction.")
