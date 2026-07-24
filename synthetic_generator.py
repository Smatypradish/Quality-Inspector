"""
Synthetic Defect Generator — Module 1
=====================================
Takes clean images from clean_images/, injects 4 types of defects
(Crack, Scratch, Rust, Dent), auto-calculates YOLO bounding box labels,
and splits them into training/validation sets.

Production Features:
  • Multi-defect injection per image (1-3 defects)
  • Augmentation pipeline (rotation, brightness, blur)
  • Progress bar with ETA
  • Robust error handling & validation
  • Reproducible seeding option
"""

import cv2
import numpy as np
import random
import os
import sys
import shutil
import argparse
import time
from pathlib import Path

# ─── Defect Class Registry ──────────────────────────────────────────────────────
CLASSES = {0: "crack", 1: "scratch", 2: "rust", 3: "dent"}

# ─── Defect Generators ──────────────────────────────────────────────────────────

def create_crack(img):
    """Generate a procedural crack defect with jagged random walk."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # Start near the center region
    curr_x = random.randint(int(w * 0.25), int(w * 0.75))
    curr_y = random.randint(int(h * 0.25), int(h * 0.75))
    points = [(curr_x, curr_y)]
    angle = random.uniform(0, 2 * np.pi)

    # Random walk with branching
    num_segments = random.randint(10, 20)
    for _ in range(num_segments):
        angle += random.uniform(-0.6, 0.6)
        length = random.randint(5, 16)
        curr_x = int(np.clip(curr_x + length * np.cos(angle), 0, w - 1))
        curr_y = int(np.clip(curr_y + length * np.sin(angle), 0, h - 1))
        points.append((curr_x, curr_y))

        # Occasional branch
        if random.random() < 0.2:
            branch_angle = angle + random.uniform(-1.2, 1.2)
            bx = int(np.clip(curr_x + 8 * np.cos(branch_angle), 0, w - 1))
            by = int(np.clip(curr_y + 8 * np.sin(branch_angle), 0, h - 1))
            cv2.line(mask, (curr_x, curr_y), (bx, by), 255, 1)

    for i in range(len(points) - 1):
        thickness = random.randint(1, 3)
        cv2.line(mask, points[i], points[i + 1], 255, thickness)

    defected = img.copy()
    defected[mask > 0] = (defected[mask > 0] * 0.2).astype(np.uint8)
    return defected, mask, 0


def create_scratch(img):
    """Generate a linear scratch defect with slight curvature."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    pt1 = (random.randint(10, w - 10), random.randint(10, h - 10))
    # Create longer scratches
    dx = random.randint(-80, 80)
    dy = random.randint(-80, 80)
    pt2 = (
        int(np.clip(pt1[0] + dx, 5, w - 5)),
        int(np.clip(pt1[1] + dy, 5, h - 5)),
    )
    thickness = random.randint(1, 2)
    cv2.line(mask, pt1, pt2, 255, thickness)

    # Add slight parallel companion line for realism
    if random.random() < 0.4:
        offset = random.choice([-2, 2])
        cv2.line(mask, (pt1[0] + offset, pt1[1] + offset),
                 (pt2[0] + offset, pt2[1] + offset), 180, 1)

    defected = img.copy()
    defected[mask > 0] = np.clip(
        defected[mask > 0].astype(int) + 120, 0, 255
    ).astype(np.uint8)
    return defected, mask, 1


def create_rust(img):
    """Generate an elliptical rust/corrosion defect with color shift."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    center = (
        random.randint(int(w * 0.15), int(w * 0.85)),
        random.randint(int(h * 0.15), int(h * 0.85)),
    )
    axes = (random.randint(12, 50), random.randint(12, 50))
    rotation = random.randint(0, 360)
    cv2.ellipse(mask, center, axes, rotation, 0, 360, 255, -1)

    # Irregular edge via noise
    noise = np.random.randint(0, 30, (h, w), dtype=np.uint8)
    mask = cv2.subtract(mask, noise)
    mask = cv2.GaussianBlur(mask, (15, 15), 0)

    defected = img.copy()
    rust_layer = defected.copy()
    rust_layer[:, :, 0] = np.clip(rust_layer[:, :, 0] * 0.25, 0, 255)   # Reduce Blue
    rust_layer[:, :, 1] = np.clip(rust_layer[:, :, 1] * 0.55, 0, 255)   # Reduce Green
    rust_layer[:, :, 2] = np.clip(rust_layer[:, :, 2] * 1.6, 0, 255)    # Boost Red

    alpha = (mask / 255.0)[:, :, None]
    defected = (defected * (1 - alpha) + rust_layer * alpha).astype(np.uint8)
    return defected, mask, 2


def create_dent(img):
    """Generate a circular dent/depression defect with shadow effect."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    center = (
        random.randint(int(w * 0.15), int(w * 0.85)),
        random.randint(int(h * 0.15), int(h * 0.85)),
    )
    radius = random.randint(10, 40)
    cv2.circle(mask, center, radius, 255, -1)
    mask = cv2.GaussianBlur(mask, (11, 11), 0)

    defected = img.copy()
    shadow = (defected * 0.35).astype(np.uint8)
    alpha = (mask / 255.0)[:, :, None]
    defected = (defected * (1 - alpha) + shadow * alpha).astype(np.uint8)
    return defected, mask, 3


# ─── Augmentation Helpers ────────────────────────────────────────────────────────

def augment_image(img):
    """Apply random augmentations: flip, brightness, slight rotation."""
    # Random horizontal flip
    if random.random() < 0.5:
        img = cv2.flip(img, 1)

    # Random brightness shift
    beta = random.randint(-25, 25)
    img = np.clip(img.astype(np.int16) + beta, 0, 255).astype(np.uint8)

    # Random slight blur
    if random.random() < 0.3:
        ksize = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)

    return img


# ─── YOLO Label Utilities ────────────────────────────────────────────────────────

def mask_to_yolo_bbox(mask, class_id):
    """Convert a binary mask to a YOLO-format bounding box string."""
    # Threshold to binary
    _, binary = cv2.threshold(mask, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    all_pts = np.vstack(contours)
    x, y, bw, bh = cv2.boundingRect(all_pts)
    h, w = mask.shape[:2]

    # Guard against zero-size boxes
    if bw < 2 or bh < 2:
        return None

    x_center = (x + bw / 2.0) / w
    y_center = (y + bh / 2.0) / h
    norm_w = bw / w
    norm_h = bh / h
    return f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n"


# ─── Progress Bar ────────────────────────────────────────────────────────────────

def print_progress(current, total, bar_len=40, prefix="Progress"):
    """Display a console progress bar."""
    fraction = current / total
    filled = int(bar_len * fraction)
    bar = "█" * filled + "░" * (bar_len - filled)
    pct = fraction * 100
    sys.stdout.write(f"\r  {prefix} |{bar}| {pct:5.1f}%  ({current}/{total})")
    sys.stdout.flush()
    if current == total:
        print()


# ─── Dataset Builder ─────────────────────────────────────────────────────────────

def build_dataset(num_samples=200, seed=None, clean_dir="clean_images", out_dir="dataset"):
    """
    Main entry point: build a synthetic YOLO training dataset.

    Args:
        num_samples: Total number of defected images to generate.
        seed:        Optional random seed for reproducibility.
        clean_dir:   Path to folder with clean reference images.
        out_dir:     Output dataset root directory.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # ── Validate clean images ────────────────────────────────────────────────
    clean_path = Path(clean_dir)
    if not clean_path.exists():
        print(f"❌ Error: Clean image directory '{clean_dir}' not found!")
        print(f"   Please create '{clean_dir}/' and add 3-5 photos of undamaged objects.")
        sys.exit(1)

    valid_ext = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    clean_files = [f for f in clean_path.iterdir() if f.suffix.lower() in valid_ext]

    if not clean_files:
        print(f"❌ Error: No valid image files found in '{clean_dir}/'!")
        print(f"   Supported formats: {', '.join(valid_ext)}")
        sys.exit(1)

    print(f"📷 Found {len(clean_files)} clean reference image(s):")
    for f in clean_files:
        print(f"   • {f.name}")

    # ── Validate images are readable ─────────────────────────────────────────
    readable_files = []
    for f in clean_files:
        test_img = cv2.imread(str(f))
        if test_img is None:
            print(f"   ⚠️  Skipping '{f.name}' — unable to decode image.")
        else:
            readable_files.append(f)

    if not readable_files:
        print("❌ Error: None of the images could be read. Check file integrity.")
        sys.exit(1)

    clean_files = readable_files

    # ── Prepare output directories ───────────────────────────────────────────
    out = Path(out_dir)
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (out / sub).mkdir(parents=True, exist_ok=True)

    generators = [create_crack, create_scratch, create_rust, create_dent]
    stats = {v: 0 for v in CLASSES.values()}
    failed = 0

    print(f"\n🚀 Generating {num_samples} synthetic defect images...\n")
    start_time = time.time()

    for i in range(num_samples):
        try:
            # Pick a random clean image
            base_file = random.choice(clean_files)
            clean_img = cv2.imread(str(base_file))

            # Apply pre-augmentation
            clean_img = augment_image(clean_img)

            # Decide how many defects (1-3) per image
            num_defects = random.choices([1, 2, 3], weights=[0.55, 0.30, 0.15])[0]
            chosen_gens = random.sample(generators, k=min(num_defects, len(generators)))

            all_labels = []
            defected_img = clean_img.copy()

            for gen_func in chosen_gens:
                defected_img, mask, class_id = gen_func(defected_img)
                bbox_str = mask_to_yolo_bbox(mask, class_id)
                if bbox_str:
                    all_labels.append(bbox_str)
                    stats[CLASSES[class_id]] += 1

            # Train/val split (80/20)
            split = "train" if random.random() < 0.8 else "val"
            filename = f"synthetic_{i:05d}"

            # Save image
            img_path = out / "images" / split / f"{filename}.jpg"
            cv2.imwrite(str(img_path), defected_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Save labels
            if all_labels:
                lbl_path = out / "labels" / split / f"{filename}.txt"
                with open(lbl_path, "w") as f:
                    f.writelines(all_labels)

        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"\n   ⚠️  Error on sample {i}: {e}")

        print_progress(i + 1, num_samples, prefix="Generating")

    elapsed = time.time() - start_time

    # ── Create data.yaml for YOLO ────────────────────────────────────────────
    abs_dataset_path = str(out.resolve()).replace("\\", "/")
    yaml_content = f"""# Auto-generated YOLO dataset config
# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
# Samples: {num_samples} | Failed: {failed}

path: {abs_dataset_path}
train: images/train
val: images/val

nc: {len(CLASSES)}
names:
  0: crack
  1: scratch
  2: rust
  3: dent
"""
    yaml_path = out / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    # ── Summary Report ───────────────────────────────────────────────────────
    train_imgs = len(list((out / "images" / "train").glob("*.jpg")))
    val_imgs = len(list((out / "images" / "val").glob("*.jpg")))

    print(f"\n{'─' * 55}")
    print(f"  ✅ Dataset Generation Complete!")
    print(f"{'─' * 55}")
    print(f"  📂 Output:       {abs_dataset_path}")
    print(f"  🖼️  Train images: {train_imgs}")
    print(f"  🖼️  Val images:   {val_imgs}")
    print(f"  ❌ Failed:       {failed}")
    print(f"  ⏱️  Time:         {elapsed:.1f}s")
    print(f"\n  📊 Defect Distribution:")
    for cls_name, count in stats.items():
        bar = "▓" * min(count // 2, 30)
        print(f"     {cls_name:>8s}: {count:4d}  {bar}")
    print(f"{'─' * 55}")


# ─── CLI Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic defect training data for YOLO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-n", "--num-samples", type=int, default=250,
        help="Number of synthetic images to generate (default: 250)",
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--clean-dir", type=str, default="clean_images",
        help="Directory containing clean reference images (default: clean_images)",
    )
    parser.add_argument(
        "--out-dir", type=str, default="dataset",
        help="Output dataset directory (default: dataset)",
    )
    args = parser.parse_args()

    print("=" * 55)
    print("  🏭 Synthetic Defect Generator v1.0")
    print("=" * 55)
    build_dataset(
        num_samples=args.num_samples,
        seed=args.seed,
        clean_dir=args.clean_dir,
        out_dir=args.out_dir,
    )
