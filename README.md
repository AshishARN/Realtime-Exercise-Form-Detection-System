# Edge AI Exercise Form Detection (Nicla Vision)

A real-time exercise form detection system running entirely on-device using the Arduino Nicla Vision.
This project demonstrates Edge AI deployment using quantized deep learning models for squat pose detection and pushup form classification.

## Features
- Real-time inference using onboard camera
- Two independent models:
1. Squat pose classification (multi-class)
2. Pushup form classification (binary: good/bad)
- INT8 quantized models for efficient embedded deployment
- On-device web dashboard (live feed + predictions)
- Dynamic model switching via browser
- Fully offline — no cloud required

## Model Details
| Model  | Architecture                             | Input Size| Classes      |
|--------|------------------------------------------|-----------|--------------|
| Squat  | MobileNetV1 0.25 (Transfer Learning)     | 96×96     | Multiple     |
| Pushup | MobileNetV1 0.25 (Transfer Learning)     | 96×96     | Good / Bad   |
Both models are trained using Edge Impulse
Quantized to INT8 for low memory and faster inference
Optimized for embedded deployment
Dataset

The dataset is a combination of:

1. Kaggle datasets
2. YouTube video frame extraction
3. Google Images
4. Manually recorded images and videos

This diverse dataset helps improve robustness across:

1. lighting conditions
2. environments
3. body variations

# System Pipeline
Capture frame from camera
Downsample (80×60 for streaming)
Resize to 96×96 for inference
Convert to model input format
Run inference (Edge Impulse SDK)
Display results on web dashboard

# Web Dashboard

The device hosts a lightweight web server:

Live grayscale camera feed
Real-time prediction label
Confidence score bars
Buttons to switch between models

Access via:

http://<device-ip>

# Hardware Requirements
Arduino Nicla Vision
WiFi connection (hotspot/router)
Power bank

#Software Requirements
Arduino IDE
Edge Impulse exported Arduino libraries
Required libraries:
WiFi.h
Camera libraries (GC2145)

# Installation
Clone the repository:
git clone https://github.com/your-username/edge-ai-exercise-detector.git
Open the .ino file in Arduino IDE
Add your WiFi credentials in:
arduino_secrets.h
Install required Edge Impulse libraries
Upload to Nicla Vision

# Usage
Power the device
Connect to WiFi
Open Serial Monitor to get device IP
Open the IP in browser
Perform exercises in front of the camera
Switch between:
Squat model
Pushup model

# Technical Highlights
Dual-model deployment on embedded hardware
Real-time inference + streaming pipeline
Efficient memory usage with quantization
Custom preprocessing pipeline (RGB565 → model input)
Lightweight HTTP server implementation

# Project Structure
.
├── arduino-final-sketch/
│   └── sketch_may01b.ino.ino
│   └── arduino_secrets.h
├── models/
│   ├── squat_model/
│   └── pushup_model/
└── README.md

# Limitations
Performance depends on lighting and camera positioning
Limited field of view (single-person detection)
Model accuracy depends on dataset diversity

# Future Work
Add more exercises (deadlift, lunges, planks, etc.)
Improve pose estimation using keypoints
Multi-person detection
Mobile app integration

# Acknowledgements
Edge Impulse for deployment tools
Open datasets from Kaggle and other sources
Arduino ecosystem for embedded support
