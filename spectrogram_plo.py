import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

# 분석할 오디오 파일 경로 (예: video (1).wav 경로)
audio_path = r"C:\Users\sophi\Desktop\archive\Coffee_room_01\Coffee_room_01\Videos\video (1).wav"

# 오디오 파일 로딩
y, sr = librosa.load(audio_path)

# 스펙트로그램 계산
D = np.abs(librosa.stft(y))  # Short-Time Fourier Transform
DB = librosa.amplitude_to_db(D, ref=np.max)

# 스펙트로그램 시각화
plt.figure(figsize=(10, 5))
librosa.display.specshow(DB, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format="%+2.0f dB")
plt.title("Spectrogram")
plt.tight_layout()
plt.show()
