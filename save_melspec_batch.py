import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 경로 설정 (상대경로 기반)
input_folder = os.path.join(".", "Coffee_room_01", "Videos")  # .wav, .avi 위치
label_folder = os.path.join(".", "Coffee_room_01", "Annotation_files")  # .txt 라벨 위치
output_folder = os.path.join(".", "output")  # fall, not_fall 폴더가 생성될 루트

# Mel spectrogram 저장 함수
def save_mel_spectrogram(audio_path, save_path):
    y, sr = librosa.load(audio_path)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# .wav 파일 반복
for file_name in os.listdir(input_folder):
    if file_name.endswith(".wav"):
        base_name = os.path.splitext(file_name)[0]  # 예: video (1)
        label_file = os.path.join(label_folder, base_name + ".txt")
        audio_path = os.path.join(input_folder, file_name)

        # 기본값: not_fall
        label = "not_fall"

        # 라벨 파일 분석
        if os.path.exists(label_file):
            with open(label_file, "r") as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split(',')
                    if len(parts) > 1 and parts[1] == "1":  # Fall 라벨
                        label = "fall"
                        break

        # 저장 경로 설정
        save_dir = os.path.join(output_folder, label)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, base_name + ".png")

        # 저장 실행
        save_mel_spectrogram(audio_path, save_path)
        print(f"저장 완료: {save_path}")
