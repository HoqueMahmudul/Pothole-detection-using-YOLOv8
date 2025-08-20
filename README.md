# Pothole Detection with YOLOv8

This repository trains a YOLOv8 model to detect potholes in road images. It assumes a single class (`pothole`) and a dataset organized in standard YOLO format with `images/` and `labels/` per split.

---

## Notebook Workflow

The notebook walks through:

1. **Setup**: install dependencies and (optionally) mount Google Drive  
2. **Dataset checks**: verify splits, labels, and quick visualizations  
3. **Config**: rewrite dataset paths to a Colab-friendly YAML (`data_colab.yaml`)  
4. **Training**: run Ultralytics training; artifacts saved to a persistent folder  
5. **Inference**: run predictions and draw boxes with confidence scores  
6. **Evaluation**: compute Precision, Recall, F1 and overlap metrics (IoU & Dice) using optimal one-to-one matching  

---

## Environment

- Python 3.9+
- Packages:
  - `ultralytics` (YOLOv8)
  - `numpy`, `pandas`, `opencv-python`, `Pillow`, `matplotlib`
  - `scipy` (Hungarian matching)
  - Optional: `roboflow` (if you fetch/convert datasets from Roboflow)

---

## Dataset

The dataset used for this project is publicly available on **Roboflow Universe**:

- **Pothole Detection YOLO-v8 Dataset** by *kartik*  
  👉 [Access Dataset on Roboflow](https://universe.roboflow.com/kartik-zvust/pothole-detection-yolo-v8/dataset/1)

### Dataset Format

Expected structure:

``` 
datasets/
└── potholesdetection-2/
    ├── data.yaml
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
```



```
class_id x_center y_center width height
```



- All coordinates are normalized to `[0, 1]`.  
- For this single-class project, `class_id` is always `0` (`pothole`).  

---

## Quickstart (Colab)

1. Open the notebook in **Google Colab**.  
2. **Runtime → Change runtime type →** select **GPU**.  
3. Run the **Setup** cell (installs packages and mounts Drive).  
4. Place your dataset at: /content/drive/MyDrive/yolov8/datasets/potholesdetection-2
5. Run the **Dataset Analysis** cells to verify counts and preview labels.  
6. Run **Config** to create `data_colab.yaml` with absolute paths.  
7. Run **Training** (start with `yolov8n.pt`, 20–30 epochs).  
8. Use **Inference** and **Evaluation** to test and measure performance.  

---

## Key Functions

- **`analyze_dataset(data_yaml)`** – loads dataset config, prints split sizes, counts images/labels.  
- **`visualize_sample_data(split="train", n=6)`** – previews images with YOLO boxes for sanity-checking.  
- **`setup_training_config(data_yaml, out_yaml)`** – rewrites dataset YAML with absolute paths (`data_colab.yaml`).  
- **`train_pothole_detector(model_name, data_yaml, epochs, imgsz, batch, device, project_dir)`** – wraps `ultralytics.YOLO(...).train(...)` and saves artifacts.  
- **`load_best_model(project_dir)` / `load_trained_model_simple(path_to_best)`** – loads `best.pt` from training outputs.  
- **`predict_potholes(model, image_path, conf_threshold=0.5)`** – runs inference and returns boxes, scores, labels.  
- **`evaluate_model_performance(model, conf_threshold=0.5, iou_threshold=0.5)`** – computes Precision, Recall, F1, and overlap metrics (IoU & Dice) via optimal one-to-one matching (Hungarian algorithm).  

---

## Evaluation & Metrics

This project evaluates the model using both **classification-style metrics** and **overlap-quality metrics**:

- **Precision**: among predicted potholes, how many are correct?  
- **Recall**: among real potholes, how many were found?  
- **F1**: harmonic mean of Precision and Recall.  
- **IoU (Jaccard Index)**:  
\[
IoU = \frac{Area\ of\ Overlap}{Area\ of\ Union}
\]  
- **Dice Coefficient**:  
\[
Dice = \frac{2 \times Area\ of\ Overlap}{Area\ of\ Prediction + Area\ of\ Ground\ Truth}
\]  

**Matching rule:** predictions and ground truth boxes are matched **one-to-one** using the Hungarian algorithm on an IoU-based cost.

The evaluator reports:
- Precision, Recall, F1  
- Mean/Median IoU and Dice  
- IoU distribution buckets (≥0.5, ≥0.75, ≥0.9)  
- Best/worst images for manual inspection  
