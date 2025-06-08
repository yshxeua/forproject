MIC_DISTANCE = 0.2  # meters
SPEED_OF_SOUND = 343  # m/s

st.title("🎤 Direction-of-Arrival Estimation (Two Stereo WAVs, Mono Channels Extracted)")
st.title("🎤 Direction-of-Arrival Estimation with Auto Mono-to-Stereo Conversion")
st.markdown("""
Upload **two stereo WAV files** recorded from microphones placed apart.
The app extracts the **left channel from the first file** and the **right channel from the second file**,
calculates the TDOA, estimates the angle of arrival, and shows:
- Waveforms of both signals
- Estimated angle in degrees
- Polar plot visualization
Upload **two WAV files** (mono or stereo). If mono, the app will convert them to stereo by duplicating the channel.
- Left channel from file 1 and right channel from file 2 are used for DoA estimation.
- Waveforms, TDOA, estimated angle, and polar plot will be displayed.
""")

file1 = st.file_uploader("Upload stereo WAV file 1 (will extract LEFT channel)", type=["wav"])
file2 = st.file_uploader("Upload stereo WAV file 2 (will extract RIGHT channel)", type=["wav"])
def ensure_stereo(data):
    """Convert mono to stereo by duplicating the channel if needed."""
    if data.ndim == 1:
        # Mono -> duplicate channel to stereo
        data = np.stack((data, data), axis=-1)
    elif data.shape[1] == 1:
        # Single channel 2D -> duplicate channel
        data = np.repeat(data, 2, axis=1)
    return data

file1 = st.file_uploader("Upload WAV file 1", type=["wav"])
file2 = st.file_uploader("Upload WAV file 2", type=["wav"])

if file1 and file2:
    sr1, data1 = wavfile.read(file1)
    sr2, data2 = wavfile.read(file2)

    if sr1 != sr2:
        st.error("⚠️ Sampling rates do not match.")
    elif data1.ndim != 2 or data1.shape[1] < 2 or data2.ndim != 2 or data2.shape[1] < 2:
        st.error("⚠️ Both files must be stereo (2 channels minimum).")
    else:
        # Extract left channel from first file and right channel from second file
        data1 = ensure_stereo(data1)
        data2 = ensure_stereo(data2)

        # Extract left channel from file1 and right channel from file2
        signal1 = data1[:, 0]
        signal2 = data2[:, 1]

        # Trim to shortest length if needed
        # Trim to shortest length
        min_len = min(len(signal1), len(signal2))
        signal1 = signal1[:min_len]
        signal2 = signal2[:min_len]

        # Normalize signals
        # Normalize
        signal1 = signal1 / np.max(np.abs(signal1))
        signal2 = signal2 / np.max(np.abs(signal2))

@@ -65,22 +73,22 @@
        tdoa = lag / sr1
        st.write(f"🕒 **TDOA**: {tdoa * 1e6:.2f} microseconds")

        # Estimate angle
        # Angle estimation
        try:
            angle_rad = asin(tdoa * SPEED_OF_SOUND / MIC_DISTANCE)
            angle_deg = degrees(angle_rad)
            st.success(f"📐 Estimated Direction of Arrival: **{angle_deg:.2f}°**")
        except ValueError:
            st.error("🚫 TDOA too large; sound source likely beyond ±90°.")
            angle_deg = None

        # Polar plot
        if angle_deg is not None:
            fig2 = plt.figure(figsize=(5, 5))
            ax = fig2.add_subplot(111, polar=True)
            ax.set_theta_zero_location('front')
            ax.set_theta_direction(-1)
            ax.plot([0, np.deg2rad(angle_deg)], [0, 1], color='magenta', linewidth=3)
            ax.set_yticklabels([])
            ax.set_title("Estimated Sound Direction", color='blue')
            st.pyplot(fig2)
