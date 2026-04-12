# Seal & Signature Detection for Legal Documents

##  Project Overview

This project builds a **computer vision system** to automatically detect **signatures and seals/stamps** in legal documents using deep learning.

### Problem Statement

Legal documents require:
- **Signatures** → proof of authorization  
- **Seals/Stamps** → institutional validation  

Manual verification is:
- Time-consuming  
- Error-prone  
- Not scalable  

This project automates the detection using **YOLOv8 object detection**.
## Working Link: https://seal-and-signature-detection.streamlit.app/

---

##  Approach Summary

The system uses:
- **YOLOv8** for object detection  
- **Synthetic dataset generation**  
- **Real-world datasets (CEDAR, Tobacco, Roboflow)**  

### Pipeline


Document (PDF/Image/DOCX)
↓
Convert to Images
↓
YOLOv8 Detection
↓
Bounding Boxes (signature, seal)
↓
Output + Visualization


---

## Data Sources & Dataset Creation

### Raw Data


data/raw/
├── tobacco/images # blank legal-style documents
├── cedar/signatures/full_org # signature images
├── roboflow/train # annotated real documents


### Dataset Details

- **Tobacco Dataset** → document backgrounds  
- **CEDAR Dataset** → signature images  
- **Roboflow Dataset** → annotated real documents  
  - Class 0 → signature  
  - Class 1 → stamp  

---

### Dataset Creation Strategy

#### 🔹 Stamp Extraction
- Extract stamps using bounding boxes from Roboflow labels  
- Save in:

data/interim/stamp_crops/


#### 🔹 Synthetic Data Generation
- Use Tobacco pages as base  
- Overlay:
  - 1 signature (from CEDAR)
  - 1 seal/stamp (from extracted crops)

#### 🔹 Final Dataset
- Combine synthetic + real images  
- Split into:
  - Train
  - Validation
  - Test  

---

## Synthetic Dataset Pipeline

Script:

dataset_generation_script.py


### Steps:
- Extract stamp crops  
- Augment signatures (resize, rotate, contrast)  
- Random placement on documents  
- Blend using alpha blending  
- Add noise and scaling  

### Output:

data/interim/stamp_crops/
data/synthetic/images/
data/synthetic/labels/


---

## Dataset Assembly

Script:

split_dataset.py


### Output Structure:

seal_dataset/
├── images/train
├── images/val
├── images/test
├── labels/train
├── labels/val
├── labels/test
└── data.yaml


### Classes:

['signature', 'seal']


---

## Training

Script:

train.py


### Configuration:
- Model: `yolov8n.pt`  
- Image Size: `640`  
- Epochs: `50`  
- Batch Size: `16`  

### Run:

python train.py


 Uses GPU if available.

---

##  Evaluation Metrics (Latest Run)

Model: **seal_signature_model4**

### Validation:
- Precision: 0.9847  
- Recall: 0.9872  
- mAP@50: 0.9911  
- mAP@50-95: 0.8637  

### Test:
- Precision: 0.979  
- Recall: 0.978  
- mAP@50: 0.990  
- mAP@50-95: 0.863  

### Per Class:

| Class      | Precision | Recall | mAP@50 | mAP@50-95 |
|-----------|----------|--------|--------|-----------|
| Signature | 0.965    | 0.955  | 0.985  | 0.786     |
| Seal      | 0.993    | 1.000  | 0.994  | 0.940     |

---

## 🌍 Real-World Inference Results

Dataset: `inference/archive (9)/1`

- 984 / 1000 → documents with detections  
- 671 → documents with signatures  
- 913 → documents with seals  
- 600 → documents with both  
- 16 → no detections  

### Observations:
- Seal detection is strong and reliable  
- Signature detection has lower confidence  
- Overlapping signature + seal reduces accuracy  

---

## 🔎 Inference

Script:

infer.py


### Supported Formats:
- JPG / PNG  
- PDF (converted via `pdftoppm`)  
- DOCX (converted via `soffice`)  

### Run:

python infer.py --source path/to/file


### Output:

runs/detect/


---

## Streamlit UI

File:

streamlit_app.py


### Features:
- Upload images, PDFs, DOCX  
- Visualize detections  
- View summary results  

### Run:

streamlit run streamlit_app.py


---

## ⚙️ How to Run (End-to-End)

### 1. Generate Dataset

python dataset_generation_script.py


### 2. Split Dataset

python split_dataset.py


### 3. Train Model

python train.py


### 4. Run Inference

python infer.py --source sample.pdf


### 5. Launch UI

streamlit run streamlit_app.py


---

## Known Limitations

- Overlapping signature and seal can cause detection errors  
- Signature detection confidence is lower than seal detection  
- Synthetic data dominates training distribution  
- No fully annotated real-world benchmark dataset  

---

## Next Steps

- Add real-world overlapping samples  
- Build annotated validation dataset  
- Improve detection for faint signatures  
- Improve robustness to noise and document artifacts  

---

## Summary

This project delivers a **complete document intelligence pipeline** using:
- YOLOv8 object detection  
- Synthetic + real dataset integration  
- Multi-format document processing  

👉 The system is **pilot-ready**, with strong seal detection and improving signat
