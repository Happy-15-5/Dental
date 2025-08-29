#Dental X-Ray Tooth Detection & FDI Numbering

This repository contains a YOLO-based object detection pipeline for detecting and numbering teeth in dental X-rays using the FDI World Dental Federation system.

The project includes:

Training YOLO on dental X-ray images

Evaluating model performance (confusion matrix, per-class metrics, loss curves)

Inference with bounding box predictions

Post-processing for anatomically correct FDI numbering

#Project Structure
├── Dental.ipynb           # Main notebook (training, evaluation, inference)
├── runs/detect/           # YOLO training outputs (weights, results, plots)
│   ├── train/weights/     # Initial training run
│   ├── train2/weights/    # Second training run
│   └── train3/weights/    # Latest training run (best.pt, last.pt)
├── data/                  # Dataset folder
│   ├── images/            # X-ray images
│   └── labels/            # YOLO annotation labels

#Setup

Clone this repo and open in Google Colab or local machine

Install dependencies:

1. pip install ultralytics
2. pip install matplotlib seaborn
3. Place dataset inside data/ in YOLO format:

data/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
#Training

Run the following in Colab or Jupyter:

from ultralytics import YOLO

model = YOLO("yolov8n.pt")   # or yolov8s.pt
model.train(
    data="data.yaml",        # dataset config file
    epochs=50,
    imgsz=640,
    batch=16,
    project="runs/detect"
)

Outputs are saved in:
runs/detect/train*/weights/best.pt

#Evaluation

Generate metrics:

results = model.val(data="data.yaml")


Confusion matrix per class

Precision/Recall/mAP per class

Training curves (loss, accuracy, precision, recall)

#Inference

Run inference on test images:

model = YOLO("runs/detect/train3/weights/best.pt")
results = model.predict(source="data/images/test", save=True, conf=0.5)


This saves predictions with bounding boxes in runs/detect/predict/.

#ost-Processing (FDI Numbering)

To ensure correct anatomical numbering, post-processing is applied:

Separate upper vs lower arch (Y-axis clustering)

Divide into left vs right quadrants (X-midline)

Sort teeth horizontally within quadrants

Assign FDI numbers sequentially

Handle missing teeth (skip numbers where spacing is wide)

#Outputs

Predicted bounding boxes with FDI IDs

Class-wise confusion matrix & metrics

Training/validation curves

#Future Work

Improve dataset size & balance

Refine post-processing heuristics

Add semi-supervised learning for better generalization
