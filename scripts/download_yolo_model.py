"""
Download a pretrained YOLOv8 model for document layout detection.

This script downloads YOLOv8n (nano) as the base model. For production,
fine-tune it using training/train_yolo.py with your labeled document dataset.

Usage:
    python scripts/download_yolo_model.py
    python scripts/download_yolo_model.py --model yolov8s  # larger/more accurate
"""

import argparse
from pathlib import Path


def download_model(model_name: str = "yolov8n", output_dir: str = "models"):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ ultralytics not installed. Run: pip install ultralytics")
        return False

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    model_file = output_path / f"{model_name}.pt"
    if model_file.exists():
        print(f"✅ Model already exists: {model_file}")
        return True

    print(f"⬇️  Downloading {model_name}.pt ...")
    model = YOLO(f"{model_name}.pt")  # auto-downloads from Ultralytics
    model_file_default = Path(f"{model_name}.pt")
    if model_file_default.exists():
        model_file_default.rename(model_file)

    print(f"✅ Saved to: {model_file}")
    print()
    print("📌 To use this model, update your .env:")
    print(f"   YOLO_MODEL_PATH=models/{model_name}.pt")
    print()
    print("⚠️  Note: This is a general-purpose model.")
    print("   For better document region detection, fine-tune with:")
    print("   python training/train_yolo.py --data dataset.yaml --epochs 50")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download YOLOv8 base model")
    parser.add_argument("--model", default="yolov8n",
                        choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l"],
                        help="Model size (n=nano, s=small, m=medium, l=large)")
    parser.add_argument("--output", default="models", help="Output directory")
    args = parser.parse_args()
    download_model(args.model, args.output)
