"""
Clean Sample Generator
======================
Generates 5 synthetic clean surface images for use as base
images in the defect injection pipeline.
"""

import cv2
import numpy as np
import os

# Create directory if it doesn't exist
os.makedirs("clean_images", exist_ok=True)

def create_brushed_metal(h=400, w=400):
    base = np.full((h, w), 190, dtype=np.float32)
    noise = np.random.normal(0, 15, (h, w)).astype(np.float32)
    # Blur horizontally to simulate brushed steel texture
    noise = cv2.GaussianBlur(noise, (1, 15), 0)
    metal = np.clip(base + noise, 0, 255).astype(np.uint8)
    return cv2.merge([metal, metal, metal])

def create_smooth_plastic(h=400, w=400):
    x = np.linspace(200, 230, w)
    y = np.linspace(200, 230, h)
    xx, yy = np.meshgrid(x, y)
    plastic = ((xx + yy) / 2).astype(np.uint8)
    return cv2.merge([plastic, plastic, plastic])

def create_cardboard_texture(h=400, w=400):
    base = np.zeros((h, w, 3), dtype=np.float32)
    base[:, :] = (160, 190, 210)  # BGR tan cardboard color
    noise = np.random.normal(0, 7, (h, w, 3))
    return np.clip(base + noise, 0, 255).astype(np.uint8)

def create_painted_sheet(h=400, w=400):
    base = np.full((h, w, 3), 235, dtype=np.float32)
    noise = np.random.normal(0, 4, (h, w, 3))
    return np.clip(base + noise, 0, 255).astype(np.uint8)

def create_matte_ceramic(h=400, w=400):
    base = np.full((h, w, 3), (220, 225, 230), dtype=np.float32)
    noise = np.random.normal(0, 3, (h, w, 3))
    return np.clip(base + noise, 0, 255).astype(np.uint8)

# Generate all 5 clean sample images
samples = {
    "clean_images/brushed_metal.jpg": create_brushed_metal(),
    "clean_images/smooth_plastic.jpg": create_smooth_plastic(),
    "clean_images/cardboard_tan.jpg": create_cardboard_texture(),
    "clean_images/painted_sheet.jpg": create_painted_sheet(),
    "clean_images/matte_ceramic.jpg": create_matte_ceramic(),
}

print("[*] Generating 5 clean base images...")
for filepath, image_data in samples.items():
    cv2.imwrite(filepath, image_data)
    print(f"  -> Created: {filepath}")

print("\n[OK] Success! 5 sample clean images generated in 'clean_images/'")
