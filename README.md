# Simple Hough Lane Detector

A compact lane detection pipeline using Canny edge detection + probabilistic Hough transform with a trapezoidal region-of-interest mask. Designed as a small, easy-to-read demo for detecting and drawing lane lines on road images.

## Features
- Grayscale + Gaussian blur preprocessing
- Canny edge detection
- Trapezoidal ROI mask to focus on road area
- Probabilistic HoughLinesP for line detection
- Averages left/right slopes to draw smooth lane lines
- Batch processing of images from `Dataset/` → outputs to `Output/`

## Requirements
- Python 3.8+
- OpenCV (`opencv-python`)
- NumPy

## Install dependencies:
```bash
pip install opencv-python numpy
```

### 1) Clone the repo:

```
git clone https://github.com/git-authority/Simple_Hough_Lane_Detector.git

cd Simple_Hough_Lane_Detector
```

### 2) Run:

```
python lane_departure_warning_system.py
```

### 3) Processed images will be saved to Output/

```
ls Output
```