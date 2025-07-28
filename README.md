# fall-detection-audio
Audio-based fall detection using Mel-spectrogram and MobileNetV2

---

## 📄 Description  
This project detects falls based on audio data. It extracts Mel-spectrograms from audio files and classifies them using a CNN (MobileNetV2).

---

## 📁 Project Structure
```
fall-detection-audio/
├── Coffee_room_01/             # Contains audio and annotation files
│   ├── Videos/                 # Input .wav files
│   └── Annotation_files/      # Ground truth labels (.txt)
├── output/                     # Output images and train folders (fall / not_fall)
├── save_melspec_image.py       # Generate a single mel-spectrogram image
├── save_melspec_batch.py       # Batch processing of mel-spectrograms
├── spectrogram_plo.py          # General spectrogram visualization
├── train_mobilenetv2.py        # Train a CNN model using MobileNetV2
├── README.md
└── LICENSE
```

---

## Dependencies
```bash
pip install librosa matplotlib numpy tensorflow
```

---

## How to Use

### 1️ 데이터 구성
```
.
├── Coffee_room_01/
│   ├── Videos/
│   │   ├── video (1).wav
│   │   └── ...
│   └── Annotation_files/
│       ├── video (1).txt
│       └── ...
```

### 2️ Mel-spectrogram 저장 (단일 파일)
```bash
python save_melspec_image.py
```

### 3️ Mel-spectrogram 저장 (일괄 처리)
```bash
python save_melspec_batch.py
```
- 결과는 `./output/fall` 또는 `./output/not_fall` 폴더에 이미지로 저장됩니다.

### 4️ 모델 학습
```bash
python train_mobilenetv2.py
```

### 5️ 스펙트로그램 시각화
```bash
python spectrogram_plo.py
```

---
