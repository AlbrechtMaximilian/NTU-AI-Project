import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import cv2  # OpenCV for resizing

def create_logmels_folder(input_folder, output_folder, sr=22050, n_mels=224, size=(224, 224)):
    # Delete existing .png images
    if os.path.exists(output_folder):
        for file in os.listdir(output_folder):
            if file.endswith(".png"):
                os.remove(os.path.join(output_folder, file))
    else:
        os.makedirs(output_folder)

    processed = 0

    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.wav'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.png')

            try:
                # Load audio
                y, sr_loaded = librosa.load(input_path, sr=sr)

                # Create log-mel spectrogram
                mel = librosa.feature.melspectrogram(y=y, sr=sr_loaded, n_mels=n_mels)
                logmel = librosa.power_to_db(mel, ref=np.max)

                # Normalize to 0–255
                logmel_norm = (logmel - logmel.min()) / (logmel.max() - logmel.min())
                logmel_img = (logmel_norm * 255).astype(np.uint8)

                # Resize and flip vertically to keep correct orientation
                resized = cv2.resize(logmel_img, size, interpolation=cv2.INTER_LINEAR)
                logmel_resized = np.flipud(resized)  # flip up-down

                # Save as image
                plt.imsave(output_path, logmel_resized, cmap='magma', format='png')

                processed += 1
                print(f"🖼 Saved log-mel: {output_path}")

            except Exception as e:
                print(f"❌ Failed on {filename}: {e}")

    print(f"\n✅ Done. {processed} spectrogram(s) saved in: {output_folder}")

# === Batch processing for all classes ===
base = "Khan Dataset"
subfolders = ["AS", "MR", "MS", "MVP", "N"]

for name in subfolders:
    filtered_path = f"Khan Dataset filtered/{name}_filtered"
    mel_output_path = f"Khan Dataset Mel/{name}"
    create_logmels_folder(filtered_path, mel_output_path, size=(224, 224))