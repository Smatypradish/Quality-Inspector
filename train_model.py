"""
YOLOv8 Model Trainer — Module 2
================================
Trains a YOLOv8 Nano model on the synthetically generated dataset.

Production Features:
  • Automatic GPU/CPU detection
  • Configurable hyperparameters via CLI
  • Pre-flight validation of dataset
  • Training progress logging
  • Post-training model evaluation summary
"""

from ultralytics import YOLO
import os
import sys
import argparse
import time
from pathlib import Path


def validate_dataset(data_yaml_path):
    """Run pre-flight checks on the dataset before training."""
    data_path = Path(data_yaml_path)

    if not data_path.exists():
        print(f"❌ Error: Dataset config not found at '{data_yaml_path}'")
        print("   Run 'python synthetic_generator.py' first to generate the dataset.")
        return False

    # Parse the YAML to check paths
    dataset_root = data_path.parent
    train_dir = dataset_root / "images" / "train"
    val_dir = dataset_root / "images" / "val"

    if not train_dir.exists():
        print(f"❌ Error: Training images directory not found: {train_dir}")
        return False

    if not val_dir.exists():
        print(f"❌ Error: Validation images directory not found: {val_dir}")
        return False

    train_count = len(list(train_dir.glob("*.jpg"))) + len(list(train_dir.glob("*.png")))
    val_count = len(list(val_dir.glob("*.jpg"))) + len(list(val_dir.glob("*.png")))

    if train_count == 0:
        print(f"❌ Error: No training images found in '{train_dir}'")
        return False

    if val_count == 0:
        print("⚠️  Warning: No validation images found. Training will proceed but metrics may be unreliable.")

    print(f"  📂 Dataset root:   {dataset_root.resolve()}")
    print(f"  🖼️  Train images:   {train_count}")
    print(f"  🖼️  Val images:     {val_count}")

    # Check label files
    train_labels = dataset_root / "labels" / "train"
    if train_labels.exists():
        label_count = len(list(train_labels.glob("*.txt")))
        print(f"  🏷️  Train labels:   {label_count}")
    else:
        print("  ⚠️  Warning: No labels directory found.")

    return True


def train(
    data_yaml="dataset/data.yaml",
    epochs=15,
    imgsz=416,
    batch=8,
    model_size="n",
    project="runs/detect",
    name="defect_inspector_model",
):
    """
    Train a YOLOv8 model on the defect detection dataset.

    Args:
        data_yaml:  Path to YOLO data.yaml config.
        epochs:     Number of training epochs.
        imgsz:      Input image resolution.
        batch:      Batch size.
        model_size: YOLO model size variant (n/s/m/l/x).
        project:    Project directory for runs.
        name:       Run name.
    """
    print("=" * 60)
    print("  🧠 YOLOv8 Defect Inspector — Training Pipeline")
    print("=" * 60)

    # ── Pre-flight checks ────────────────────────────────────────────────
    print("\n📋 Pre-flight Dataset Validation:")
    if not validate_dataset(data_yaml):
        sys.exit(1)

    # ── Detect compute device ────────────────────────────────────────────
    try:
        import torch
        if torch.cuda.is_available():
            device = "0"
            gpu_name = torch.cuda.get_device_name(0)
            print(f"\n  🚀 GPU Detected: {gpu_name}")
        else:
            device = "cpu"
            print("\n  💻 No GPU detected — training on CPU (this may be slow).")
    except ImportError:
        device = "cpu"
        print("\n  💻 PyTorch not available for GPU check — using CPU.")

    # ── Initialize model ─────────────────────────────────────────────────
    model_weights = f"yolov8{model_size}.pt"
    print(f"\n  📦 Loading pre-trained weights: {model_weights}")

    try:
        model = YOLO(model_weights)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)

    # ── Training configuration summary ───────────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"  ⚙️  Training Configuration:")
    print(f"     Epochs:      {epochs}")
    print(f"     Image Size:  {imgsz}px")
    print(f"     Batch Size:  {batch}")
    print(f"     Device:      {device}")
    print(f"     Model:       YOLOv8{model_size.upper()}")
    print(f"     Project:     {project}/{name}")
    print(f"{'─' * 60}")

    # ── Start training ───────────────────────────────────────────────────
    print("\n⚡ Starting Training...\n")
    start_time = time.time()

    try:
        results = model.train(
            data=os.path.abspath(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            name=name,
            project=project,
            exist_ok=True,
            patience=10,           # Early stopping patience
            save=True,
            save_period=5,         # Checkpoint every 5 epochs
            plots=True,            # Generate training plots
            verbose=True,
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user. Partial weights may have been saved.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        sys.exit(1)

    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    # ── Post-training summary ────────────────────────────────────────────
    best_weights = Path(project) / name / "weights" / "best.pt"
    last_weights = Path(project) / name / "weights" / "last.pt"

    print(f"\n{'=' * 60}")
    print(f"  🎉 Training Complete!")
    print(f"{'=' * 60}")
    print(f"  ⏱️  Duration:    {minutes}m {seconds}s")

    if best_weights.exists():
        size_mb = best_weights.stat().st_size / (1024 * 1024)
        print(f"  📦 Best model:  {best_weights}  ({size_mb:.1f} MB)")
    else:
        print(f"  ⚠️  Best weights not found at expected path.")

    if last_weights.exists():
        print(f"  📦 Last model:  {last_weights}")

    print(f"\n  💡 Next step: Launch the dashboard with:")
    print(f"     streamlit run app.py")
    print(f"{'=' * 60}")


# ─── CLI Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 model for visual defect detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data", type=str, default="dataset/data.yaml",
        help="Path to YOLO data.yaml (default: dataset/data.yaml)",
    )
    parser.add_argument(
        "--epochs", type=int, default=15,
        help="Number of training epochs (default: 15)",
    )
    parser.add_argument(
        "--imgsz", type=int, default=416,
        help="Training image resolution (default: 416)",
    )
    parser.add_argument(
        "--batch", type=int, default=8,
        help="Batch size (default: 8)",
    )
    parser.add_argument(
        "--model", type=str, default="n", choices=["n", "s", "m", "l", "x"],
        help="YOLOv8 model size: n(ano)/s(mall)/m(edium)/l(arge)/x(tra-large) (default: n)",
    )
    parser.add_argument(
        "--name", type=str, default="defect_inspector_model",
        help="Run name (default: defect_inspector_model)",
    )
    args = parser.parse_args()

    train(
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        model_size=args.model,
        name=args.name,
    )
