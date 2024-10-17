# Drone Surveillance System

## Overview

This project uses YOLOv8 for real-time object detection from various video sources (webcam, local files, YouTube) with a PyQt5-based GUI. Detected objects are annotated on the video stream.

## Features

- **Multiple Video Sources**: Webcam, local video files, YouTube.
- **Real-Time Object Detection**: Powered by YOLOv8.
- **User-Friendly GUI**: Built with PyQt5.
- **Error Handling**: Displays and logs errors related to video sources and model processing.

## Requirements

- Python 3.x
- `opencv-python`, `PyQt5`, `numpy`, `yt-dlp`, `ultralytics`

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/shahinur-alam/Drone-Surveillance-System.git
   cd Drone-Surveillance-System
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Download the YOLOv8 model (yolov8n.pt) if necessary.

## How to Run

1. Run the application:

   ```bash
   python drone_surveillance.py
   ```
2. Select a video source (Camera, Video File, or YouTube URL).
3. Click Start to begin and Stop to end the video feed.

## Key Components
- YOLO Model: YOLOv8 is loaded for object detection.
- Video Source: Select from webcam, local files, or YouTube URLs.
- Real-Time Annotations: Detected objects are displayed on the video feed.
- Error Handling: Errors are displayed and logged.

## Future Improvements
- Support for custom YOLO models.
- Option to save video with annotations.
- More robust YouTube stream handling.
