# Audio Anomaly Detection and Source Separation using Deep Learning

Audio processing project focused on human voice detection and audio source separation using Deep Learning architectures based on U-Net and BiLSTM networks.

Developed as part of the **Artificial Intelligence and Machine Learning** course within the M.Sc. in Data Science and Innovation Management at the University of Salerno.

---

## Project Overview

The project addresses two closely related tasks in real-world audio processing:

### Task 1 — Audio Anomaly Detection

Development of a non-supervised Deep Learning system capable of detecting the presence of human voices within noisy environmental audio recordings.

The model learns the acoustic characteristics of background sounds and automatically identifies voice events as anomalies.

### Task 2 — Audio Source Separation

Development of a supervised audio separation architecture capable of isolating human speech from environmental noise.

The system reconstructs two independent audio streams:

- Voice component
- Background component

This allows cleaner speech extraction and improved intelligibility in noisy environments.

---

## Motivation

Human voice detection and separation are fundamental tasks in modern Artificial Intelligence systems.

Potential applications include:

- Speech enhancement in noisy environments
- Video conferencing systems
- Automatic speech recognition pipelines
- Environmental acoustic monitoring
- Smart assistants
- Public safety and emergency detection systems
- Audio surveillance and monitoring

---

## Datasets

The project combines public datasets containing speech and environmental sounds.

### Voice Dataset

- Mozilla Common Voice

### Environmental Audio Datasets

- ESC-50
- UrbanSound8K

Audio samples were filtered to obtain clips between 5 and 9 seconds in duration in order to ensure dataset consistency.

---

## Data Preparation Pipeline

The dataset preparation process included:

1. Collection of voice and background recordings
2. Dataset splitting:
   - Training (80%)
   - Validation (10%)
   - Test (10%)
3. Generation of synthetic audio mixtures
4. Creation of:
   - Voice tracks
   - Background tracks
   - Mixed audio tracks
5. Export of dataset metadata into CSV files

Each sample contains:

```text
background.wav
voice.wav
mixture.wav
```

---

## Task 1: Audio Anomaly Detection

### Objective

Detect the presence of human speech within environmental audio recordings.

### Learning Strategy

The model is trained only on background audio.

During inference:

- Background sounds are considered normal patterns
- Human speech is treated as an anomaly

### Architecture

Custom 1D U-Net architecture featuring:

- Encoder-Decoder structure
- 5 hierarchical levels
- Residual connections
- Skip connections
- BiLSTM temporal modeling
- Dual output branches

Total parameters:

```text
11,530,947
```

### Output

The network produces:

- Voice component
- Background component

allowing anomaly localization and interpretation.

---

## Task 2: Audio Source Separation

### Objective

Separate speech signals from environmental noise.

### Architecture

Wave-U-Net inspired architecture composed of:

- Encoder
- Bottleneck
- Decoder
- Skip Connections
- BiLSTM layers

Activation functions:

- LeakyReLU
- GELU
- Tanh

Total parameters:

```text
10,892,246
```

### Training Strategy

The supervised model receives:

```text
Input:
Mixture Audio

Targets:
Voice Track
Background Track
```

The model learns to reconstruct both components simultaneously.

---

## Audio Preprocessing

Audio preprocessing was performed using Librosa and PyTorch.

Main operations:

- Mono conversion
- Resampling to 16 kHz
- Padding and truncation
- Amplitude normalization
- RMS normalization
- Data augmentation

The preprocessing pipeline improves training stability and model robustness.

---

## Training Configuration

### Loss Function

L1 Loss

Advantages:

- Robust against outliers
- Stable convergence
- Effective for waveform reconstruction

### Optimizer

AdamW

Advantages:

- Better generalization
- Reduced overfitting
- Stable optimization process

### Monitoring

- Validation Loss
- Checkpoint Saving
- Reconstruction Quality
- Logging Metrics

Best validation loss achieved:

```text
0.00021
```

---

## Evaluation Metrics

The models were evaluated using:

- SDR (Signal-to-Distortion Ratio)
- SNR (Signal-to-Noise Ratio)
- Pearson Correlation
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)

---

## Experimental Results

### Anomaly Detection

Results obtained on unseen test samples:

| Metric | Value |
|----------|----------|
| Processed Files | 15,643 |
| Mean Error | 0.0307 |
| Standard Deviation | 0.0279 |
| Minimum Error | 0.0000 |
| Maximum Error | 0.1746 |

The model demonstrated good generalization capability and stable anomaly detection performance.

---

### Audio Source Separation

| Metric | Event |
|----------|----------|
| SDR | 14.27 |
| Correlation | 0.94 |
| MAE | 0.0170 |
| MSE | 0.00168 |

| Metric | Background |
|----------|----------|
| SDR | 6.03 |
| Correlation | 0.82 |
| MAE | 0.0170 |
| MSE | 0.00162 |

The model achieved effective voice reconstruction and satisfactory background separation.

---

## Whisper Evaluation

The reconstructed speech signals were additionally evaluated using OpenAI Whisper.

The transcription quality demonstrated that the separated speech signals preserved the linguistic content with high accuracy, confirming the effectiveness of the source separation pipeline.

---

## Repository Contents

This repository contains the academic documentation of the project, including:

- Project presentation
- Methodology description
- Experimental results
- Model architecture overview

The original source code developed during the course is no longer available.

---

## Technologies

- Python
- PyTorch
- Librosa
- NumPy
- Deep Learning
- U-Net
- Wave-U-Net
- BiLSTM
- Audio Signal Processing
- OpenAI Whisper

---

## Academic Context

Course:

**Artificial Intelligence and Machine Learning**

M.Sc. in Data Science and Innovation Management

University of Salerno

---

## Authors

**Giuseppe Rega**  
**Asia Bruno**  
**Felicita Giliberti**  
**Pasquale Langelotti**