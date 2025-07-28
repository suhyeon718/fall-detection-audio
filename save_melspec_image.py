import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

def save_mel_spectrogram(audio_path, output_path):
    y, sr = librosa.load(audio_path)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(4, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# 예시 실행
save_mel_spectrogram(
    os.path.join(".", "Coffee_room_01", "Videos", "video (1).wav"),
    os.path.join(".", "output", "fall_01.png")
)

