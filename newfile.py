import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from math import asin, degrees
from numpy.fft import fft, ifft
from scipy.signal import windows

# Constants
MIC_DISTANCE = 0.2  # meters
SPEED_OF_SOUND = 343  # m/s

st.title("🎯 Direction-of-Arrival Estimation Using Microphone Array")
st.markdown("""
Objective: 
Estimate the direction from which a sound is coming using time delays between microphones. 
Instructions: 

Set up two microphones spaced apart. 
Record a clap or sound from various angles. 

Use cross-correlation to calculate time difference of arrival (TDOA). 

Calculate the angle of arrival based on microphone distance and TDOA. 

Display results in degrees or on a polar plot. 
""")

def ensure_stereo(data):
    """Convert mono to stereo by duplicating the channel if needed."""
    if data.ndim == 1:
        return np.stack((data, data), axis=-1)
    elif data.shape[1] == 1:
        return np.repeat(data, 2, axis=1)
    return data

def gcc_phat(sig, refsig, fs, interp=16):
    """Compute GCC-PHAT cross-correlation and TDOA."""
    n = sig.shape[0] + refsig.shape[0]
    SIG = fft(sig, n=n*interp)
    REFSIG = fft(refsig, n=n*interp)
    R = SIG * np.conj(REFSIG)
    R = R / np.abs(R + 1e-15)  # Avoid divide by zero
    cc = np.real(ifft(R))
    max_shift = int(interp * MIC_DISTANCE / SPEED_OF_SOUND * fs)
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
    shift = np.argmax(cc) - max_shift
    tdoa = shift / float(fs * interp)
    return tdoa, cc

file1 = st.file_uploader("Upload WAV file 1", type=["wav"])
file2 = st.file_uploader("Upload WAV file 2", type=["wav"])

if file1 and file2:
    sr1, data1 = wavfile.read(file1)
    sr2, data2 = wavfile.read(file2)

    if sr1 != sr2:
        st.error("⚠️ Sampling rates do not match.")
    else:
        fs = sr1
        data1 = ensure_stereo(data1)
        data2 = ensure_stereo(data2)

        # Convert to float and zero-mean
        sig1 = data1[:, 0].astype(np.float64)
        sig2 = data2[:, 1].astype(np.float64)
        min_len = min(len(sig1), len(sig2))
        sig1, sig2 = sig1[:min_len], sig2[:min_len]
        sig1 -= np.mean(sig1)
        sig2 -= np.mean(sig2)
        sig1 /= np.max(np.abs(sig1))
        sig2 /= np.max(np.abs(sig2))

        # Apply windowing
        window = windows.hann(min_len)
        sig1 *= window
        sig2 *= window

        # Plot signals
        time = np.linspace(0, min_len / fs, min_len)
        fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axs[0].plot(time, sig1, color='blue')
        axs[0].set_title("Left Channel (File 1)")
        axs[0].set_ylabel("Amplitude")
        axs[0].grid(True)
        axs[1].plot(time, sig2, color='red')
        axs[1].set_title("Right Channel (File 2)")
        axs[1].set_ylabel("Amplitude")
        axs[1].set_xlabel("Time [s]")
        axs[1].grid(True)
        st.pyplot(fig)

        # Compute TDOA using GCC-PHAT
        tdoa, _ = gcc_phat(sig1, sig2, fs)
        st.write(f"🕒 **GCC-PHAT TDOA**: {tdoa * 1e6:.2f} microseconds")

        # DoA Estimation
        try:
            angle_rad = asin(tdoa * SPEED_OF_SOUND / MIC_DISTANCE)
            angle_deg = degrees(angle_rad)
            st.success(f"📐 Estimated DoA: **{angle_deg:.2f}°**")
        except ValueError:
            st.error("🚫 TDOA too large; sound source likely beyond ±90°.")
            angle_deg = None
        # Polar plot
        if angle_deg is not None:
            fig2 = plt.figure(figsize=(5, 5))
            ax = fig2.add_subplot(111, polar=True)
            ax.set_theta_zero_location('N')  # 'N' = North (top of plot)
            ax.set_theta_direction(-1)       # Clockwise
            ax.plot([0, np.deg2rad(angle_deg)], [0, 1], color='magenta', linewidth=3)
            ax.set_yticklabels([])
            ax.set_title("Estimated Sound Direction", color='blue')
            st.pyplot(fig2)

            # Optional: also show Left/Right info
            if angle_deg < 0:
                st.info("🔊 Source is to the **Left**")
            elif angle_deg > 0:
                st.info("🔊 Source is to the **Right**")
            else:
                st.info("🔊 Source is **Straight Ahead**")

