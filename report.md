# Edge AI-Based Real-Time Exercise Form Detection System (Push-ups + Squats)

> **Course:** Edge AI [CP 330]
>
> **Team:** Anjesh (MTech CSA, IISc Bangalore) · Ashish Nambiar (MTech CSA, IISc Bangalore) · Garima Papnai (MTech AI, IISc Bangalore) · Shubham Bijalwan (MTech Smart Manufacturing, IISc Bangalore)

---

## 1. Problem Statement, Motivation & Objectives

Fitness form mistakes during push-ups and squats are a leading cause of workout-related injuries and reduced training effectiveness. Real-time feedback typically requires gym trainers or expensive cloud-connected systems — neither of which is practical for everyday home use. The goal of this project is to build a **low-cost, portable, fully offline** solution that runs entirely on an embedded device.

This project develops a real-time, on-device exercise form detection system using Edge AI on the Arduino Nicla Vision. Two lightweight MobileNetV1 models — one for push-ups and one for squats — are trained via transfer learning, quantized to INT8, and deployed directly on the microcontroller. Classification results are served through an on-device web dashboard accessible from any phone on the local network.

**Why Edge AI?**
- Real-time inference with minimal latency (42–53 ms per frame on-device)
- Privacy-preserving: no video or image data leaves the device
- No internet dependency — fully offline operation
- Low-cost and portable: runs on a ₹5000-range embedded board

**Key Objectives:**
- Collect and label a custom image dataset for push-up and squat form classification
- Train lightweight MobileNetV1-based models using transfer learning via Edge Impulse
- Apply INT8 quantization to optimize models for constrained hardware
- Deploy models on Arduino Nicla Vision (STM32H747) using the EON Compiler
- Build an on-device web dashboard for live feedback with model switching

---

## 2. Proposed Solution (Overview)

The system implements a complete Edge AI pipeline — from image capture to on-device classification and live feedback:

```
Camera Capture → Resize 96×96 → RGB565 → EI Format (float buffer)
                                               ↓
                                    Model Selection (Squat / Pushup)
                                               ↓
                                    Run Inference (INT8 Model)
                                               ↓
                              Prediction Output (Label + Confidence)
                                               ↓
                    ┌──────────────────────────────────────────────┐
                    ↓                                              ↓
         Downsample 80×60 (Streaming)               On-device Web Server
                                                           ↓
                                               Web Dashboard (Live Feed + Scores)
```

The Arduino Nicla Vision captures frames, converts them to the Edge Impulse float buffer format, and runs inference locally using the selected INT8 model. A dual-model system allows switching between push-up and squat classifiers via the web dashboard. Results — label and confidence scores — are streamed to a phone browser in real time over the local network.

| Model | Task | Classes |
|-------|------|---------|
| Push-up | Binary classification | `bad` (incorrect form), `good` (correct form) |
| Squat | Multi-class classification | `badform`, `deep`, `shallow` |

---

## 3. Hardware & Software Setup

**Hardware:**

| Component | Details |
|-----------|---------|
| Board | Arduino Nicla Vision |
| MCU | STM32H747 (Cortex-M7 @ 480 MHz + Cortex-M4) |
| Camera | Built-in onboard camera (captures exercise frames) |
| Memory | 1 MB RAM, 2 MB Flash |
| Power | Power bank (portable operation) |

**Software:**

| Tool | Purpose |
|------|---------|
| Edge Impulse | Dataset management, model training, deployment pipeline |
| TensorFlow Lite Micro | On-device inference runtime |
| EON Compiler (RAM optimized) | Model optimization and memory-efficient deployment |
| OpenMV IDE | Firmware scripting, camera integration, and flashing on Nicla Vision |
| Arduino IDE | Firmware (.ino) development and flashing (alternative deployment path) |
| Google Colab | Burst-frame extraction pipeline (video → labeled image dataset) |

---

## 4. Data Collection & Dataset Preparation

The dataset is a combination of multiple sources to improve robustness across lighting conditions, environments, and body variations:

- **Kaggle datasets** — [https://www.kaggle.com/code/youssefemad004/pushups-data-videopreprocssing-data](https://www.kaggle.com/code/youssefemad004/pushups-data-videopreprocssing-data)
- **Zenodo squat dataset** — Teng, C. (2025). Squat Dataset [Data set]. Zenodo. [https://doi.org/10.5281/zenodo.17558630](https://doi.org/10.5281/zenodo.17558630)
- **YouTube video frame extraction** — [https://www.youtube.com/watch?v=txnwoJz-Rno](https://www.youtube.com/watch?v=txnwoJz-Rno), [https://www.youtube.com/watch?v=daDK0huWvfc](https://www.youtube.com/watch?v=daDK0huWvfc)
- **Google Images** — supplementary frames for class diversity
- **Self-recorded images and videos** — captured using the Nicla Vision onboard camera, placed on the floor pointing at the person performing exercises (side/front view). Frames were extracted using a custom burst-frame extraction pipeline (Google Colab notebook) and labeled manually via the Edge Impulse Data Acquisition interface.

### Push-up Dataset

Side-view push-up videos (correct and incorrect form) were recorded and converted to labeled image frames. The dataset was split at the **video level** to prevent data leakage between train and test sets.

| Split | Bad (incorrect) | Good (correct) | Total |
|-------|:--------------:|:--------------:|:-----:|
| Train | 1,509 | 1,062 | 2,571 |
| Validation | 431 | 303 | 734 |
| Test | 216 | 153 | 369 |
| **Total** | **2,156** | **1,518** | **3,674** |

> **Class imbalance:** Bad-form samples outnumber good-form by ~42%. Data augmentation was enabled during training to compensate. Per-class F1 reflects this: bad (0.88) vs good (0.83).

### Squat Dataset

| Detail | Value |
|--------|-------|
| Total samples | 509 images |
| Classes | `badform`, `deep`, `shallow` |
| Train / Test split | 75% / 25% |
| Source | Self-collected + Kaggle + Zenodo (Teng 2025) + YouTube frames |

### Preprocessing Steps

- Images resized to **96×96 pixels** using "fit shortest axis" mode
- Frames extracted from videos via Colab burst-frame pipeline
- RGB565 → float buffer conversion handled in firmware at inference time
- Manual labeling performed in Edge Impulse Data Acquisition
- Noisy and ambiguous samples removed during review
- 100% of data subset used for training

### Edge Impulse Impulse Design

| Block | Push-up Config | Squat Config |
|-------|---------------|-------------|
| Input | 96×96 image, fit shortest axis | 96×96 image, fit shortest axis |
| Processing | Image block | Image block |
| Learning | Transfer Learning (MobileNetV1) | Transfer Learning (MobileNetV1) |
| Output features | 2 classes: bad, good | 3 classes: badform, deep, shallow |

---

## 5. Model Design, Training & Evaluation

Both models use **MobileNetV1** as the backbone with transfer learning (ImageNet pretrained weights fine-tuned on exercise data), implemented via Edge Impulse's Transfer Learning block.

---

### Push-up Model

**Training Configuration:**

| Parameter | Value |
|-----------|-------|
| Architecture | MobileNetV1 96×96 0.1 (final layer: 8 neurons, 0.1 dropout) |
| Input features | 27,648 |
| Epochs | 50 |
| Learning rate | 0.0005 |
| Data augmentation | Enabled |
| Training processor | CPU |

> Note: Initial training used a larger MobileNetV1 architecture which caused Flash overflow on Nicla Vision. Architecture was reduced to MobileNetV1 0.1 to fit within device memory.

**Results (Validation Set — Quantized INT8):**

| Metric | Value |
|--------|-------|
| Accuracy | **85.9%** |
| Loss | 0.30 |
| ROC-AUC | 0.86 |
| Weighted Precision | 0.87 |
| Weighted Recall | 0.86 |
| Weighted F1 Score | 0.86 |

**Confusion Matrix:**

| | Predicted: BAD | Predicted: GOOD |
|--|:-:|:-:|
| Actual: BAD | **84.4%** | 15.6% |
| Actual: GOOD | 11.8% | **88.2%** |
| F1 Score | 0.88 | 0.83 |

**Interpretation:** Strong binary classification with clear visual separation between correct and incorrect push-up postures. The slightly lower good-form F1 (0.83 vs 0.88) is consistent with the dataset class imbalance. Data augmentation was enabled to partially compensate.

---

### Squat Model

**Training Configuration:**

| Parameter | Value |
|-----------|-------|
| Architecture | MobileNetV1 96×96 0.25 (no final dense layer, 0.1 dropout) |
| Input features | 9,216 |
| Epochs | 80 |
| Learning rate | 0.0004 |
| Data augmentation | Disabled |
| Training processor | CPU |

**Results (Validation Set — Quantized INT8):**

| Metric | Value |
|--------|-------|
| Accuracy | **79.2%** |
| Loss | 0.48 |
| ROC-AUC | 0.93 |
| Weighted Precision | 0.80 |
| Weighted Recall | 0.79 |
| Weighted F1 Score | 0.80 |

**Confusion Matrix:**

| | Predicted: BADFORM | Predicted: DEEP | Predicted: SHALLOW |
|--|:-:|:-:|:-:|
| Actual: BADFORM | **84.8%** | 15.2% | 0% |
| Actual: DEEP | 5% | **70%** | 25% |
| Actual: SHALLOW | 4.2% | 16.7% | **79.2%** |
| F1 Score | 0.89 | 0.65 | 0.79 |

**Interpretation:** Badform detection is strong (F1: 0.89). The main confusion is between deep and shallow squats (25% of deep misclassified as shallow), expected due to high visual similarity at intermediate squat depths. The high ROC-AUC (0.93) confirms strong overall discriminative ability.

---

### Model Comparison Summary

| Model | Task | Classes | Accuracy | Weighted F1 | ROC-AUC |
|-------|------|:-------:|:--------:|:-----------:|:-------:|
| Push-up | Binary | 2 | **85.9%** | 0.86 | 0.86 |
| Squat | Multi-class | 3 | **79.2%** | 0.80 | 0.93 |

---

## 6. Model Compression & Efficiency Metrics

**Technique: INT8 Post-Training Quantization** via Edge Impulse EON Compiler (RAM-optimized)

All weights and activations are quantized from 32-bit float to 8-bit integer, enabling faster fixed-point arithmetic on the Cortex-M7 and significantly reducing memory footprint.

### Compression Results (target: Arduino Nicla Vision, Cortex-M7 @ 480 MHz)

| Metric | Float32 (Unoptimized) | INT8 (Quantized) | Improvement |
|--------|:--------------------:|:----------------:|:-----------:|
| Total Latency | 141 ms | **54 ms** | ~2.6× faster |
| Peak RAM | 300.1 KB | **124.7 KB** | ~58% reduction |
| Flash | 857.5 KB | **284.0 KB** | ~67% reduction |

**Key Observations:**
- ~2.6× faster inference enables real-time operation
- ~58% RAM reduction brings both models within Nicla Vision's 1 MB limit
- ~67% Flash reduction leaves headroom for firmware and web server
- Negligible accuracy loss between INT8 and Float32 on validation set
- EON Compiler further reduces footprint beyond standard TFLite quantization

---

## 7. Model Deployment & On-Device Performance

**Deployment Steps:**
1. Train and validate models on Edge Impulse Studio (cloud)
2. Select INT8 quantization + EON Compiler (RAM-optimized) in deployment settings
3. Export as **OpenMV library** (`.zip`) targeting Arduino Nicla Vision
4. Flash firmware to Nicla Vision using **OpenMV IDE** (primary) or **Arduino IDE** via `.ino` file
5. Arduino `.ino` firmware handles: image capture → preprocessing → inference → web server output
6. Add WiFi credentials in `arduino_secrets.h` and connect device to hotspot/router
7. Results streamed to **on-device web dashboard** — access via the IP shown in Serial Monitor from any phone browser on the local network

**On-Device Performance (EON Compiler, RAM-optimized):**

| Model | Inference Time | Peak RAM | Flash Usage |
|-------|:-------------:|:--------:|:-----------:|
| Push-up (INT8, EON) | **42 ms** | 77.7 KB | 106.3 KB |
| Squat (INT8, EON) | **53 ms** | 115.9 KB | 295.8 KB |

Both models run comfortably within Nicla Vision hardware limits (1 MB RAM, 2 MB Flash). At 42–53 ms per inference, the system achieves approximately **19–24 classifications per second** — sufficient for real-time exercise monitoring.

---

## 8. System Prototype

### System Pipeline

The full end-to-end pipeline implemented in firmware:

```
Camera Capture (Nicla Vision)
        ↓
  Resize 96×96 (Inference)
        ↓
  RGB565 → EI Format (float buffer)
        ↓                          ↘
  Model Selection              Downsample 80×60 (Streaming)
  (Squat / Pushup)
        ↓
  Run Inference (INT8 Model)
        ↓
  Prediction Output (Label + Confidence)
        ↓
  On-device Web Server
        ↓
  Web Dashboard (Live Feed + Scores)
```

### On-Device Web Dashboard

The firmware hosts a lightweight web server accessible from any phone browser on the local network. The dashboard ("Exercise Classifier") provides:
- **Model toggle buttons**: Switch between Squat and Pushup classifier in real time
- **Active model label**: Shows which model is currently running
- **Live camera feed**: Downsampled 80×60 grayscale stream from Nicla Vision
- **Classification output**: Current label (good/bad/badform/deep/shallow) displayed prominently
- **Confidence bars**: Per-class confidence scores shown as progress bars

### Live Demo

The system was demonstrated live with a person performing push-ups in front of the Nicla Vision (placed on the floor). The phone browser showed real-time classification with confidence scores updating with each inference cycle. A demo video is available in the repository.

---

## 9. Conclusions & Limitations

**Conclusions:**
- Successfully deployed a real-time Edge AI exercise form detection system on the Arduino Nicla Vision with no cloud dependency
- Push-up binary classifier achieved **85.9% accuracy** with strong bad/good form separation
- Squat multi-class classifier achieved **79.2% accuracy** with excellent badform detection (F1: 0.89) and high discriminative ability (ROC-AUC: 0.93)
- INT8 quantization reduced inference latency by **~2.6×** and RAM by **~58%** with negligible accuracy loss
- On-device web dashboard enables live feedback accessible from any phone — no app install needed

**Limitations:**
- Sensitive to lighting conditions — performance degrades in low or inconsistent light
- Single-person detection only — system not designed for multi-person scenes
- Accuracy depends heavily on dataset quality and camera angle consistency
- No temporal modeling — each frame classified independently, ignoring motion context
- Deep vs. shallow squat confusion (25%) due to high visual similarity at intermediate depths
- Limited environmental diversity in training data (backgrounds, body types, angles)

---

## 10. Future Work

- Add more exercises (lunges, planks, deadlifts, burpees)
- Use **pose estimation (keypoints)** — e.g., MediaPipe Pose — for skeleton-based form analysis
- Add **multi-person support** for group fitness settings
- **Mobile app integration** for a richer user experience beyond the browser dashboard
- Add **repetition counting** using temporal state tracking
- Expand dataset with diverse users, lighting, and camera angles
- Explore temporal models (TCN, LSTM) using IMU data for motion-aware classification

---

## 11. Challenges & Mitigation

| Challenge | Impact | Mitigation Applied |
|-----------|--------|--------------------|
| Video frame extraction created near-duplicate images | Risk of data leakage between train/test | Split performed at video level, not frame level |
| Class imbalance in push-up dataset (bad >> good) | Biased model, lower good-form F1 | Data augmentation enabled; per-class F1 monitored |
| Initial larger model caused Flash overflow on Nicla Vision | Deployment failure | Switched to MobileNetV1 0.1; applied INT8 + EON Compiler |
| High inference latency on Float32 model (141 ms) | Not real-time capable | INT8 quantization brought latency to 42–53 ms |
| Deep vs. shallow squat class overlap | 25% misclassification | Increased epochs to 80; acknowledged as fundamental visual ambiguity |
| Sensitivity to lighting during live demo | Inconsistent inference results | Controlled demo environment; noted as key limitation for future work |

---

## 12. References

1. **MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications** — Howard et al., 2017 — https://arxiv.org/abs/1704.04861
2. **TensorFlow Lite Micro** — https://www.tensorflow.org/lite/microcontrollers
3. **Edge Impulse Documentation** — https://docs.edgeimpulse.com
4. **Arduino Nicla Vision Documentation** — https://docs.arduino.cc/hardware/nicla-vision/
5. **OpenMV IDE** — https://openmv.io/
6. **Push-up Dataset (Kaggle)** — https://www.kaggle.com/code/youssefemad004/pushups-data-videopreprocssing-data
7. **Squat Dataset (Zenodo)** — Teng, C. (2025). Squat Dataset [Data set]. Zenodo. https://doi.org/10.5281/zenodo.17558630
