import json
import os
import shutil
from glob import glob
from sklearn.model_selection import train_test_split
from pathlib import Path

# === CONFIG ===
BASE_DIR = Path(__file__).parent.parent.parent
DATASET_DIR = BASE_DIR / "data" /  "raw"  # carpeta donde están las imágenes y los JSON
OUTPUT_DIR = BASE_DIR / "data" / "yolo"
CLASS_NAME = "uva_bbox"          # tu clase Supervisely

# ================================

def convert_bbox(x1, y1, x2, y2, img_w, img_h):
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return cx, cy, w, h


def process_split(dataset_dir, output_dir, input_split, output_split):
    img_dir = os.path.join(dataset_dir, input_split, "img")
    ann_dir = os.path.join(dataset_dir, input_split, "ann")

    output_img_dir = os.path.join(output_dir, "images", output_split)
    output_lbl_dir = os.path.join(output_dir, "labels", output_split)

    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_lbl_dir, exist_ok=True)

    image_files = sorted(glob(os.path.join(img_dir, "*.jpeg")))

    print(f"[{input_split} → {output_split}] {len(image_files)} imágenes.")

    for img_path in image_files:
        base = os.path.splitext(os.path.basename(img_path))[0]
        json_path = os.path.join(ann_dir, base + ".jpeg.json")

        if not os.path.exists(json_path):
            print(f"⚠ No existe JSON para {img_path}. Saltando.")
            continue

        with open(json_path, "r") as f:
            data = json.load(f)

        img_w = data["size"]["width"]
        img_h = data["size"]["height"]
        label_out_path = os.path.join(output_lbl_dir, base + ".txt")

        with open(label_out_path, "w") as out:
            for obj in data["objects"]:
                if obj["geometryType"] != "rectangle":
                    continue
                if obj["classTitle"] != CLASS_NAME:
                    continue

                (x1, y1), (x2, y2) = obj["points"]["exterior"]
                cx, cy, w, h = convert_bbox(x1, y1, x2, y2, img_w, img_h)
                out.write(f"0 {cx} {cy} {w} {h}\n")

        shutil.copy(img_path, output_img_dir)


# Crear carpetas base
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "labels"), exist_ok=True)

# Procesar splits
process_split(dataset_dir=DATASET_DIR, output_dir=OUTPUT_DIR, input_split="train", output_split="train")
process_split(dataset_dir=DATASET_DIR, output_dir=OUTPUT_DIR, input_split="test", output_split="val")

print("Conversión completada con carpetas estándar YOLO.")