# 🔏 Seal & Signature Detector

> **YOLOv8-based detection of handwritten signatures and official seals/stamps in document images**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)](https://github.com/ultralytics/ultralytics)
[![mAP@50](https://img.shields.io/badge/mAP%4050-97.1%25-brightgreen)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey)]()

---

## Overview

This project trains and deploys a real-time object detection model to locate two key document elements in scanned or photographed document images:

| Class ID | Label | Description |
|----------|-------|-------------|
| `0` | `signature` | Handwritten cursive/semi-cursive marks |
| `1` | `seal` | Official stamps, ink seals, or embossed marks |

The model is built on **YOLOv8n** (Ultralytics) and trained on a **hybrid dataset** of 4,902 labeled document images combining synthetic generation and real-world annotation.

---

## Results

| Metric | Final Epoch (50) | Best Epoch (48) |
|--------|-----------------|-----------------|
| Precision | **0.9545** | 0.9481 |
| Recall | **0.9260** | 0.9315 |
| mAP@50 | **0.9710** | **0.9712** |
| mAP@50-95 | **0.7646** | 0.7661 |

Training converged stably with near-zero gap between training and validation losses, confirming strong generalization without overfitting.

---

## Dataset

The training dataset is a **hybrid collection** combining synthetic and real-world document images.

### Composition

| Source | Images | Notes |
|--------|--------|-------|
| Synthetic (generated) | 3,000 | Built on real document layouts |
| Real — Legacy Roboflow | 297 | Older manually labeled collection |
| Real — Roboflow2 | 1,605 | Newer, larger, consistently labeled |
| **Total** | **4,902** | |

### Splits

| Split | Synthetic | Legacy Real | New Real | Total |
|-------|-----------|-------------|----------|-------|
| Train | 2,100 | 207 | 1,156 | **3,463** |
| Validation | 599 | 60 | 289 | **948** |
| Test | 301 | 30 | 160 | **491** |

### Instance Counts (Training Split)

- **Signatures:** 3,970 instances
- **Seals:** 3,764 instances
- **Class balance:** ~51% / 49% — near-perfect balance

### Image Patterns (Training Split)

| Pattern | Count | % |
|---------|-------|----|
| Both signature + seal | 2,892 | 83.5% |
| Signature only | 399 | 11.5% |
| Seal only | 116 | 3.3% |
| Neither (negative) | 56 | 1.6% |

---

## Project Structure

```
seal-signature-detector/

├──dataset_generation_script.py # helps in generating the synthetic part of the dataset
├── split_dataset.py            # Builds final dataset from 3 sources
├── train.py                    # Training entry point
├── infer.py                # Run inference on new images
├── runs/
│   └── detect/
│       └── seal_signature_model2/   # Training outputs
│           ├── weights/
│               ├── best.pt
│           
├──streamlit_app.py        # UI of the project  
├──requirements.txt
└── README.md
```

---

## Quickstart

### 1. Install Dependencies

```bash
pip install ultralytics
```

### 2. Prepare the Dataset

```bash
python split_dataset.py
```

This script merges the three data sources (synthetic, legacy real, new real) and writes train/val/test splits to `seal_dataset/`.

### 3. Train

```bash
yolo detect train \
  model=yolov8n.pt \
  data=seal_dataset/data.yaml \
  epochs=50 \
  imgsz=640 \
  batch=16 \
  name=seal_signature_model2 \
  device=0
```

Or using Python:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data="seal_dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="seal_signature_model2",
    device=0
)
```

### 4. Evaluate

```bash
yolo detect val \
  model=runs/detect/seal_signature_model2/weights/best.pt \
  data=seal_dataset/data.yaml
```

### 5. Run Inference

```bash
yolo detect predict \
  model=runs/detect/seal_signature_model2/weights/best.pt \
  source=path/to/your/document.jpg \
  imgsz=640 \
  conf=0.25
```

Or in Python:

```python
from ultralytics import YOLO

model = YOLO("runs/detect/seal_signature_model2/weights/best.pt")
results = model.predict("document.jpg", imgsz=640, conf=0.25)

for r in results:
    for box in r.boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        label = "signature" if cls == 0 else "seal"
        print(f"Detected: {label} ({conf:.2f})")
```

---

## Dataset YAML Format

```yaml
# seal_dataset/data.yaml
path: ./seal_dataset
train: images/train
val: images/val
test: images/test

nc: 2
names:
  0: signature
  1: seal
```

---

## Label Format

All annotations follow YOLO format — one `.txt` file per image with one detection per line:

```
<class_id> <x_center> <y_center> <width> <height>
```

All values are normalized to `[0, 1]` relative to image dimensions. Example:

```
0 0.512 0.743 0.231 0.087
1 0.308 0.614 0.195 0.201
```

---

## Training Configuration Summary

| Parameter | Value |
|-----------|-------|
| Model | YOLOv8n |
| Pretrained weights | `yolov8n.pt` (COCO) |
| Input size | 640 × 640 px |
| Batch size | 16 |
| Epochs | 50 |
| Device | GPU (device 0) |
| Run name | `seal_signature_model2` |
| Dataset config | `seal_dataset/data.yaml` |

---

## Final Loss Values (Epoch 50)

| Loss | Train | Validation |
|------|-------|------------|
| Box Loss | 0.8006 | 0.8072 |
| Class Loss | 0.4238 | 0.3951 |
| DFL Loss | 1.0740 | 1.0785 |

The near-identical train/val losses confirm stable convergence without significant overfitting.

---

## Known Limitations

- **Seal-only images are underrepresented** (3.3% of training) — isolated seal detection may be slightly weaker than co-occurrence detection.
- **Negative images are sparse** (1.6%) — consider adding more clean background images to reduce false positives in deployment.
- **mAP@50-95 gap** — bounding box localization on irregular signature shapes still has room to improve.

---


## License

This project is released under the [MIT License](LICENSE).
