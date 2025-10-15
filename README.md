# Real-time Face Mask Detector (OpenCV + MobileNetV2)

Detects whether people are wearing face masks in real-time using a webcam. When someone without a mask (or below a coverage threshold) is detected, the system can save a screenshot of the face and send an email alert to an administrator.

This repository includes scripts for: dataset preparation, training a MobileNetV2-based binary mask classifier, and a real-time OpenCV interface that visualizes mask coverage and sends alerts.

---

## Project components

* `prepare_dataset.py` / `split_mask_dataset.py` — scripts to prepare and organize datasets into the layout expected by the trainer (`dataset/train/*`, `dataset/val/*`).
* `train_mask_detector.py` — trains a MobileNetV2-based binary classifier (mask / no_mask) using Keras/TensorFlow and saves a `.h5` model.
* `realtime_mask_interface.py` — real-time webcam interface using OpenCV: detects faces, runs the classifier, displays percentage coverage, overlays colored highlights (green = masked, red = unmasked), and sends email alerts with face screenshots (rate-limited).
* `requirements.txt` — required Python packages (create one from the venv using `pip freeze` or use the example below).

---

## Research / Reference

This project is inspired by and similar to academic and practical works on real-time mask detection. A related paper (used for reference) is available here:

Real-Time Face Mask Detection and Email Alert: [https://ijsrem.com/download/real-time-face-mask-detection-and-email-alert/](https://ijsrem.com/download/real-time-face-mask-detection-and-email-alert/)

---

## Quickstart (recommended)

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <repo-folder>
```

### 2. Create a Python virtual environment (recommended)

```bash
cd "<path-to-project>"
python3 -m venv venv
source venv/bin/activate       # macOS / Linux (zsh/bash)
# Windows (PowerShell): .\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

**Example requirements (if `requirements.txt` not provided):**

```
numpy
opencv-python
pillow
scikit-learn
matplotlib
tensorflow   # or tensorflow-macos + tensorflow-metal for Apple Silicon
```

> Choose `tensorflow-macos` and `tensorflow-metal` on Apple Silicon (M1/M2) if you want hardware-accelerated training/inference.

---

## Prepare dataset (manual download + auto-organization)

1. Manually download a face-mask dataset (e.g., AndrewMVD Face Mask Detection from Kaggle, MaskedFace-Net, RMFD, MAFA) and extract it to a folder.
2. If you downloaded the dataset manually, place it under `raw_data/` or any folder and edit the `split_mask_dataset.py` to point to the correct `ANNOTATIONS_DIR` and `IMAGES_DIR`.
3. Run the split script to create `dataset/train` and `dataset/val`:

```bash
python3 split_mask_dataset.py
```

After running, the expected structure is:

```
dataset/
  train/
    mask/
    no_mask/
  val/
    mask/
    no_mask/
```

Notes:

* The included `split_mask_dataset.py` supports datasets with Pascal VOC-style XML annotations (Kaggle AndrewMVD variant). If your dataset uses a different format, either adapt the script or tell me which dataset you downloaded and I can provide a parser.
* Optional: include `incorrect_mask` images as `no_mask` or as a separate class if you want 3-class training (requires modifying the training script).

---

## Train the model

Basic training command:

```bash
python3 train_mask_detector.py --data_dir dataset --epochs 12 --batch_size 32 --output mask_detector_mobilenetv2_binary.h5
```

Important notes:

* The training script uses Transfer Learning with `MobileNetV2` (ImageNet weights) and trains a single sigmoid-output head for binary classification.
* Check the printed `train_gen.class_indices` to confirm folder-to-label mapping (it should show `{'mask': 0, 'no_mask': 1}` or similar depending on folder names).
* For imbalanced datasets: use class weights or oversampling.
* Use callbacks such as `ModelCheckpoint`, `ReduceLROnPlateau`, and `EarlyStopping` (already included in the script template).

---

## Run real-time detection

Requirements:

* `mask_detector_mobilenetv2_binary.h5` saved in the project root (or update the path in `realtime_mask_interface.py`).
* OpenCV face detector files: `deploy.prototxt.txt` and `res10_300x300_ssd_iter_140000.caffemodel` placed in project root.

Set email environment variables (optional):

```bash
export ALERT_EMAIL="youremail@gmail.com"
export ALERT_EMAIL_PWD="your_smtp_app_password"
export ALERT_RECIPIENT="admin@example.com"
```

Run:

```bash
python3 realtime_mask_interface.py
```

The webcam will open. The UI displays: per-face coverage percentage, colored overlay per slice of the face (green for masked slices, red for unmasked), label text, and bounding boxes. If coverage falls below the configured threshold, a screenshot will be saved in `alerts/` and an email alert will be attempted.

Press `q` to quit.

---

## Configuration & parameters

Edit `realtime_mask_interface.py` to tweak:

* `CONF_THRESHOLD` — face detector confidence threshold.
* `SLICE_COUNT` — how many horizontal slices to split the face into for coverage estimation.
* `COVERAGE_ALERT_THRESHOLD` — coverage fraction below which an alert is raised.
* `ALERT_COOLDOWN` — seconds to wait before sending another alert for a nearby detection.

Edit `train_mask_detector.py` to change training hyperparameters (epochs, batch size, learning rate) or to switch to 3-class training.

---

## Improving accuracy & robustness

* Combine multiple datasets (Kaggle + MaskedFace-Net + RMFD + MAFA) for greater variety.
* Add augmentation (lighting, rotation, occlusion) during training.
* If you need pixel-level mask segmentation, train a lightweight segmentation model (U-Net / DeepLab) on annotated mask bitmaps.
* For deployment on edge devices, convert the model to TFLite or ONNX and quantize for speed.

---

## Privacy & ethics

* Only use this system in compliance with local privacy laws and institutional policies.
* Store images responsibly and purge old alerts when possible.
* Inform people if you collect or store identifiable images.

---

## Troubleshooting

* "cv2 not found": ensure you're in the virtual environment and `opencv-python` is installed.
* "OOM (Out of memory) during training": reduce `--batch_size`, use fewer `--epochs`, or train on a machine with GPU.
* Email sending fails: verify `ALERT_EMAIL` and `ALERT_EMAIL_PWD` (Gmail requires App Passwords if 2FA is enabled), network access, and that SMTP settings match your provider.

---

## License

MIT

---

## Acknowledgements

* The dataset(s) used in this project are public. See dataset sources in the project or the README dataset section.
* Reference paper: *Real-Time Face Mask Detection and Email Alert* (IJSREM). Link: [https://ijsrem.com/download/real-time-face-mask-detection-and-email-alert/](https://ijsrem.com/download/real-time-face-mask-detection-and-email-alert/)
