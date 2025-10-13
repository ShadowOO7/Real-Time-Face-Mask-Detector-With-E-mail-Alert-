# New Initial Project

This repository was reset to start from a clean initial state. See `START_HERE.md` for the step-by-step process for this reset and how to upload your new project files.
# Real-time Face Mask Coverage Detector (OpenCV + MobileNetV2)

A real-time face mask detector and *approximate coverage* visualizer built with OpenCV and TensorFlow/Keras.
- Opens your laptop camera automatically.
- Detects faces and predicts mask/no-mask.
- Estimates % of face covered by the mask using a lightweight slice-based heuristic.
- Highlights masked regions in **green**, unmasked in **red**.
- Sends email alerts with face screenshot when coverage is below a configurable threshold (rate-limited).

## Files
- `train_mask_detector.py` — training script using MobileNetV2 for binary mask/no_mask classification.
- `realtime_mask_interface.py` — real-time webcam UI, coverage visualization, and email alerts.
- `deploy.prototxt.txt` & `res10_300x300_ssd_iter_140000.caffemodel` — OpenCV DNN face detector (download separately).
- `mask_detector_mobilenetv2_binary.h5` — trained model file (created after training).
- `README.md` — this file.

## How to prepare/organize datasets for train_mask_detector.py
Your training script expects this structure:
    dataset/
        train/
            mask/
            no_mask/
        val/
            mask/
            no_mask/


## Requirements
- Python 3.8+
- TensorFlow 2.x
- OpenCV (cv2)
- numpy
- pillow
- (optional) imutils

Install dependencies:
```bash
pip install tensorflow opencv-python numpy pillow