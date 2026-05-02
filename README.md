# Edge AI Exercise Form Detection (Nicla Vision)
Course: Edge AI(CP 330)
Course Website Link:https://www.samy101.com/edge-ai-26/
Video Presentation Link : edge-ai-project-demo.mp4 (https://indianinstituteofscience-my.sharepoint.com/:v:/g/personal/nashish_iisc_ac_in/IQAqXLfEHrliRLEZ4zchh9PXAa7CAM8ltmTAOALboT3e4LQ?e=AYBbFi&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifX0%3D)

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
| Pushup | MobileNetV1  0.1 (Transfer Learning)     | 96×96     | Good / Bad   |

Both models are trained using Edge Impulse
Quantized to INT8 for low memory and faster inference
Optimized for embedded deployment
Dataset

The dataset is a combination of:

1. Kaggle datasets (https://www.kaggle.com/code/youssefemad004/pushups-data-videopreprocssing-data)
2. Teng, C. (2025). Squat Dataset [Data set]. Zenodo. https://doi.org/10.5281/zenodo.17558630
2. YouTube video frame extraction (https://www.youtube.com/watch?v=txnwoJz-Rno, https://www.youtube.com/watch?v=daDK0huWvfc)
3. Google Images
4. Manually recorded images and videos

This diverse dataset helps improve robustness across:

1. lighting conditions
2. environments
3. body variations

# System Pipeline
* Capture frame from camera
* Downsample (80×60 for streaming)
* Resize to 96×96 for inference
* Convert to model input format
* Run inference (Edge Impulse SDK)
* Display results on web dashboard


# Web Dashboard

* The device hosts a lightweight web server:
* Live grayscale camera feed
* Real-time prediction label
* Confidence score bars
* Buttons to switch between models
* Access via: http://<device-ip>  // can be modified in the .ino file

# Hardware Requirements
* Arduino Nicla Vision
* WiFi connection (hotspot/router)
* Power bank

# Software Requirements
* Arduino IDE
* Edge Impulse exported Arduino libraries

# Required libraries:
* WiFi.h
* Camera libraries (GC2145)

# Installation
1. Clone the repository:
2. git clone https://github.com/your-username/edge-ai-exercise-detector.git
3. Open the .ino file in Arduino IDE
4. Add your WiFi credentials in: arduino_secrets.h
5. Install required Edge Impulse libraries
6. Upload to Nicla Vision

# Usage
1. Power the device
2. Connect to WiFi
3. Open Serial Monitor to get device IP
4. Open the IP in browser
5. Perform exercises in front of the camera
6. Switch between:
* Squat model
* Pushup model

# Technical Highlights
1. Dual-model deployment on embedded hardware
2. Real-time inference + streaming pipeline
3. Efficient memory usage with quantization
4. Custom preprocessing pipeline (RGB565 → model input)
5. Lightweight HTTP server implementation


# Limitations
* Performance depends on lighting and camera positioning
* Limited field of view (single-person detection)
* Model accuracy depends on dataset diversity

# Future Work
* Add more exercises (deadlift, lunges, planks, etc.)
* Improve pose estimation using keypoints
* Multi-person detection
* Mobile app integration

# Acknowledgements
* Edge Impulse for deployment tools
* Open datasets from Kaggle and other sources
* Arduino ecosystem for embedded support
