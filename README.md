# 🛡️ AI Visual Quality Inspector

Real-time defect detection system powered by **YOLOv8** and **Streamlit**. Inspects objects via webcam, classifies surface defects (Crack, Scratch, Rust, Dent), and renders a live **PASS / FAIL** verdict on a sleek industrial dashboard.

---

## Features

- **Synthetic Dataset Generator** — Procedurally injects cracks, scratches, rust spots, and dents onto clean images with auto-generated YOLO labels.
- **YOLOv8 Training Pipeline** — One-command model training with pre-flight dataset validation and early stopping.
- **Real-Time Dashboard** — Modern dark-slate Streamlit UI with:
  - Live camera feed with bounding box overlays
  - Animated PASS (green) / FAIL (red pulsing) badges
  - FPS, latency, and frame-count metrics
  - Defect history table with CSV export
  - Settings panel for confidence threshold and camera selection

---

## Project Structure

```
defect_inspector/
├── clean_images/              # Drop 3-5 photos of undamaged objects here
├── dataset/                   # Auto-generated YOLO training dataset
│   ├── images/train/ & val/
│   ├── labels/train/ & val/
│   └── data.yaml
├── generate_clean_samples.py  # Creates synthetic clean textures
├── synthetic_generator.py     # Injects defects + generates labels
├── train_model.py             # YOLOv8 training script
├── app.py                     # Streamlit dashboard (main entry)
├── requirements.txt           # Python dependencies
└── .gitignore
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Clean Samples (optional)

```bash
python generate_clean_samples.py
```

### 3. Generate Synthetic Defect Dataset

```bash
python synthetic_generator.py
```

### 4. Train the YOLOv8 Model

```bash
python train_model.py
```

After training, copy `best.pt` to the project root:

```bash
copy runs\detect\train\weights\best.pt best.pt
```

### 5. Launch the Dashboard

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Object Detection | YOLOv8 (Ultralytics) |
| Computer Vision | OpenCV |
| Dashboard UI | Streamlit + Custom CSS |
| Language | Python 3.10+ |

---

## Screenshots

---

## License

MIT License — free for personal and commercial use.
