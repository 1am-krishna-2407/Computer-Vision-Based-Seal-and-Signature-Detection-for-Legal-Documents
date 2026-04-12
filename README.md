Seal & Signature Detection for Legal Documents
🚀 Project Overview

This project presents a computer vision pipeline for detecting signatures and seals/stamps in legal documents using deep learning.

🎯 Problem Statement

Legal documents often require:

Signatures → proof of authorization
Seals/Stamps → institutional validation

Manual verification is:

time-consuming
error-prone
difficult at scale

👉 This project automates the detection process using object detection models.

💡 Approach Summary

The system uses:

YOLOv8 for object detection
A combination of:
synthetic data (generated)
real-world datasets

Pipeline:

Document (PDF/Image/DOCX)
        ↓
Preprocessing (conversion to images)
        ↓
YOLOv8 Detection
        ↓
Bounding Boxes (signature, seal)
        ↓
Visualization + Output
📊 Data Sources & Dataset Creation
📁 Raw Data Sources
data/raw/
├── tobacco/images                # blank legal-style documents
├── cedar/signatures/full_org     # signature images
├── roboflow/train                # annotated real documents
📌 Dataset Details
Tobacco Dataset → document backgrounds
CEDAR Dataset → signature samples
Roboflow Dataset → real annotated documents
Class 0 → signature
Class 1 → stamp
🧱 Dataset Creation Strategy
🔹 Stamp Extraction
Bounding boxes from Roboflow annotations are used
Cropped stamps are saved into:
data/interim/stamp_crops/
🔹 Synthetic Composition
Blank document pages (Tobacco) are used as base
Each page is composited with:
1 signature
1 seal/stamp
🔹 Final Dataset
Synthetic + real images are merged
Dataset is split into:
Train / Validation / Test
🏗 Synthetic Dataset Pipeline

Main script:

dataset_generation_script.py
🔧 Pipeline Steps
Stamp Crop Extraction
Extract stamps using Roboflow bounding boxes
Signature Augmentation
Resize, rotate, adjust contrast
Placement Strategy
Random placement on document regions
Avoid unrealistic overlaps
Blending
Alpha blending to simulate ink
Randomization
Scale, position, noise
📂 Output Directories
data/interim/stamp_crops/
data/synthetic/images/
data/synthetic/labels/
📦 Dataset Assembly

Script:

split_dataset.py
📁 Output Structure
seal_dataset/
├── images/
│   ├── train/
│   ├── val/
│   ├── test/
├── labels/
│   ├── train/
│   ├── val/
│   ├── test/
└── data.yaml
🏷 Classes
names: ['signature', 'seal']
🧠 Training

Script:

train.py
⚙️ Configuration
Model: yolov8n.pt
Image size: 640
Epochs: 50
Batch size: 16
python train.py

👉 Automatically uses GPU if available, otherwise CPU.

📈 Evaluation Metrics (Latest Run)

Model: seal_signature_model4

🔹 Validation (Final Epoch)
Precision: 0.9847
Recall: 0.9872
mAP@50: 0.9911
mAP@50-95: 0.8637
🔹 Test Split (Held-Out)
Precision: 0.979
Recall: 0.978
mAP@50: 0.990
mAP@50-95: 0.863
Per Class:
Class	Precision	Recall	mAP@50	mAP@50-95
Signature	0.965	0.955	0.985	0.786
Seal	0.993	1.000	0.994	0.940
🌍 Real-World Inference Check

Dataset:

inference/archive (9)/1
📊 Summary
984 / 1000 → documents with detections
671 → with signatures
913 → with seals
600 → with both
16 → no detections
📌 Observations
✅ Seal detection is high confidence and stable
⚠️ Signature detection shows lower confidence
❗ Known issue:
Overlapping signature + seal can reduce accuracy
🔎 Inference

Script:

infer.py
📥 Supported Formats
JPG / PNG
PDF
DOCX
🔄 Conversion Pipeline
PDF → Images via pdftoppm
DOCX → PDF via soffice
▶️ Run Inference
python infer.py --source path/to/file
📤 Output
runs/detect/

Includes:

annotated images
bounding boxes
🖥 Streamlit UI

File:

streamlit_app.py
✨ Features
Upload:
Images
PDFs
DOCX
Visual detection output
Summary of detections
▶️ Run UI
streamlit run streamlit_app.py
⚙️ How to Run (End-to-End)
1️⃣ Generate Dataset
python dataset_generation_script.py
2️⃣ Split Dataset
python split_dataset.py
3️⃣ Train Model
python train.py
4️⃣ Run Inference
python infer.py --source sample.pdf
5️⃣ Launch UI
streamlit run streamlit_app.py
#Summary
This project demonstrates a complete document intelligence pipeline combining:

synthetic data generation
real dataset integration
YOLOv8 detection
multi-format inference
