import os
import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import butter, lfilter

def butter_lowpass(cutoff, fs, order=6):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_lowpass_filter(data, fs, cutoff=600, order=6):
    b, a = butter_lowpass(cutoff, fs, order)
    return lfilter(b, a, data)

def filter_lowpass_folder(input_folder, output_folder, cutoff=600, order=6):
    os.makedirs(output_folder, exist_ok=True)
    processed = 0

    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.wav'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            try:
                fs, data = wav.read(input_path)

                if len(data.shape) == 2:  # stereo → mono
                    data = np.mean(data, axis=1)

                filtered = apply_lowpass_filter(data, fs, cutoff=cutoff, order=order)
                filtered = filtered / np.max(np.abs(filtered))
                filtered_int16 = np.int16(filtered * 32767)

                wav.write(output_path, fs, filtered_int16)
                processed += 1
                print(f"✔ Filtered: {filename}")

            except Exception as e:
                print(f"❌ Failed on {filename}: {e}")

    print(f"\n✅ Done. Filtered {processed} file(s) into: {output_folder}")

# Input folders
base = "Khan Dataset"
subfolders = ["AS", "MR", "MS", "MVP", "N"]

# Step 1: Filter audio files
for name in subfolders:
    input_path = os.path.join(base, name)
    filtered_path = f"Khan Dataset filtered/{name}_filtered"
    filter_lowpass_folder(input_path, filtered_path, cutoff=600, order=6)