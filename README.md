# Real-Time Face Anti-Spoofing System

## Overview
This project is a professional YOLO-based real-time face anti-spoofing and image quality assessment system. It is designed to differentiate between real and spoofed faces (e.g., printed images, videos, masks) with high precision. The system integrates a Streamlit GUI for user interaction and provides real-time performance.

## Features
- **YOLOv8-based Detection**: Trained model for precise face detection and anti-spoofing.
- **Image Quality Assessment**: Blurring detection using Laplacian variance for improved results.
- **Streamlit Interface**: User-friendly interface with adjustable confidence thresholds and real-time webcam feed display.
- **Custom Dataset Support**: Includes scripts for data collection, labeling, and splitting into train/val/test sets.
- **Stop Button**: Allows easy termination of the webcam feed.

## Project Architecture
- **Frontend**: Built using Streamlit for visualization and user interaction.
- **Backend**: Python-based processing with YOLO model integration, OpenCV for image handling, and data processing scripts.
- **Model**: Trained YOLOv8 model (`latestversion.pt`) for detecting real vs fake faces.

## System Requirements
### Hardware
- Minimum: Core i5 Processor, 8GB RAM, Integrated Webcam
- Recommended: Core i7 Processor, 16GB RAM, NVIDIA GPU (for model training)

### Software
- Python 3.9+
- Libraries: `opencv-python`, `streamlit`, `ultralytics`, `cvzone`

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/real-time-face-anti-spoofing.git
   cd real-time-face-anti-spoofing


├── Dataset/
│   ├── Datacollect/         # Stores captured data
│   ├── SplitData/           # Train/Val/Test data splits
├── Models/
│   ├── latestversion.pt     # Trained YOLO model
├── Scripts/
│   ├── datacollection.py    # Data collection script
│   ├── splitdata.py         # Data splitting script
│   ├── train.py             # YOLO model training script
├── app.py                   # Streamlit app for real-time detection
├── main.py                  # OpenCV-based standalone detection
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation


   ```plaintext
   opencv-python
   streamlit
   ultralytics
   cvzone

##Output
Real_Time_Face_Anti-spoofing_Detection\Outputrealandfake.png
