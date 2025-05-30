import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import correlate
from math import asin, degrees
import os

# Constants
MIC_DISTANCE = 0.2  # Distance between microphones in meters
SPEED_OF_SOUND = 343  # Speed of sound in m/s

def estimate_doa(wav_path):
    if not os.path.exists(wav_path):
        print("⚠️ File does not exist. Please check the path.")
        return

    # Load audio file
    sample_rate, data = wavfile.read(wav_path)
    
    if data.ndim != 2 or data.shape[1] != 2:
        print("⚠️ This is not a stereo file. Please record with two microphones.")
        return

    left = data[:, 0]
    right = data[:, 1]

    # Normalize
    left = left / np.max(np.abs(left))
    right = right / np.max(np.abs(right))

    # Cross-correlation
    corr = correlate(left, right, mode='full')
    lags = np.arange(-len(left) + 1, len(left))
    lag = lags[np.argmax(corr)]
    tdoa = lag / sample_rate  # Time Difference of Arrival in seconds

    print(f"🕒 TDOA: {tdoa * 1e6:.2f} microseconds")

    # Calculate angle
    try:
        angle_rad = asin(tdoa * SPEED_OF_SOUND / MIC_DISTANCE)
        angle_deg = degrees(angle_rad)
        print(f"📐 Estimated Direction of Arrival: {angle_deg:.2f}°")
    except ValueError:
        print("🚫 Angle is out of detectable range. Sound may have arrived outside of ±90°.")
        return

    # Plot polar
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_zero_location('front')
    ax.set_theta_direction(-1)
    ax.plot([0, np.deg2rad(angle_deg)], [0, 1], color='magenta', linewidth=3)
    ax.set_yticklabels([])
    ax.set_title("Estimated Direction of Arrival", fontsize=14, color='blue')
    plt.show()


# === USAGE ===
# Replace 'your_audio.wav' with the path to your stereo WAV file
estimate_doa("your_audio.wav")
