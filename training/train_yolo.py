"""
YOLOv8 Fine-tuning Script for Document Region Detection
Trains on custom document layout dataset (DocLayNet / custom annotations).

Usage:
    python training/train_yolo.py --data data.yaml --epochs 50 --model yolov8n.pt
    python training/train_yolo.py --resume runs/train/exp/weights/last.pt
"""

import argparse
import yaml
import shutil
from pathlib import Path
from loguru import logger


# ─── Dataset YAML generator ───────────────────────────────────────────────────

def create_dataset_yaml(
    train_dir: str,
    val_dir: str,
    test_dir: str = None,
    output_path: str = "data.yaml",
):
    """Generate a YOLO-format dataset YAML config."""
    labels = [
        "text_block", "table", "figure", "signature",
        "stamp", "logo", "header", "footer",
        "checkbox", "form_field", "barcode", "qrcode",
    ]
    data = {
        "path": str(Path(train_dir).parent.absolute()),
        "train": train_dir,
        "val": val_dir,
        "nc": len(labels),
        "names": labels,
    }
    if test_dir:
        data["test"] = test_dir

    with open(output_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

    logger.info(f"Dataset YAML written to: {output_path}")
    return output_path


# ─── Trainer ─────────────────────────────────────────────────────────────────

class DocumentYOLOTrainer:
    """
    Wraps Ultralytics YOLOv8 training with document-specific
    augmentation settings and callbacks.
    """

    # Augmentation preset optimized for documents
    # (no heavy color jitter — documents are mostly black/white)
    DOCUMENT_AUGMENTATION = {
        "hsv_h": 0.01,       # minimal hue shift
        "hsv_s": 0.1,        # minimal saturation
        "hsv_v": 0.3,        # some brightness variation (scanner differences)
        "degrees": 5.0,      # small rotation (slight skew)
        "translate": 0.05,
        "scale": 0.2,
        "shear": 2.0,
        "perspective": 0.0,
        "flipud": 0.0,       # NO vertical flip (documents have orientation)
        "fliplr": 0.0,       # NO horizontal flip
        "mosaic": 0.5,
        "mixup": 0.0,
    }

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        output_dir: str = "runs/train",
        device: str = "cpu",
    ):
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.device = device
        self._model = None

    def load_model(self):
        from ultralytics import YOLO
        self._model = YOLO(self.model_path)
        logger.info(f"Model loaded: {self.model_path}")

    def train(
        self,
        data_yaml: str,
        epochs: int = 50,
        imgsz: int = 1024,        # Higher res for document details
        batch: int = 8,
        lr0: float = 0.001,
        patience: int = 15,       # Early stopping
        pretrained: bool = True,
        project_name: str = "doc_cv",
        experiment_name: str = "train",
        workers: int = 4,
        amp: bool = False,        # Mixed precision (disable for CPU)
    ):
        """
        Run YOLOv8 training with document-optimized settings.

        Args:
            data_yaml:       Path to dataset YAML
            epochs:          Total training epochs
            imgsz:           Input image size (1024 recommended for docs)
            batch:           Batch size
            lr0:             Initial learning rate
            patience:        Early stopping patience (epochs without improvement)
            pretrained:      Use COCO pretrained weights
            project_name:    Output directory name
            experiment_name: Run name
        """
        if self._model is None:
            self.load_model()

        logger.info(f"Starting training: {epochs} epochs, imgsz={imgsz}")

        results = self._model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            lr0=lr0,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            patience=patience,
            device=self.device,
            project=project_name,
            name=experiment_name,
            pretrained=pretrained,
            workers=workers,
            amp=amp,
            save=True,
            save_period=10,
            val=True,
            plots=True,
            verbose=True,
            **self.DOCUMENT_AUGMENTATION,
        )
        logger.success(f"Training complete. Results saved to: {results.save_dir}")
        return results

    def validate(self, data_yaml: str, weights_path: str):
        """Run validation on trained model."""
        from ultralytics import YOLO
        model = YOLO(weights_path)
        metrics = model.val(data=data_yaml, imgsz=1024, verbose=True)
        logger.info(f"mAP50: {metrics.box.map50:.4f}")
        logger.info(f"mAP50-95: {metrics.box.map:.4f}")
        return metrics

    def export(self, weights_path: str, format: str = "onnx", imgsz: int = 1024):
        """
        Export trained weights to ONNX / TorchScript / TFLite for deployment.
        Supported: onnx, torchscript, tflite, coreml, engine (TensorRT)
        """
        from ultralytics import YOLO
        model = YOLO(weights_path)
        path = model.export(format=format, imgsz=imgsz, simplify=True)
        logger.success(f"Model exported to: {path}")
        return path


# ─── Annotation helpers ───────────────────────────────────────────────────────

def convert_coco_to_yolo(coco_json: str, output_dir: str, image_dir: str):
    """
    Convert COCO-format annotations (from LabelStudio / CVAT) to YOLO txt format.
    Each image gets a .txt file with: class_id cx cy w h (normalized)
    """
    import json
    import os

    with open(coco_json) as f:
        coco = json.load(f)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Build id → filename map
    id_to_image = {img["id"]: img for img in coco["images"]}
    # Build category id → index map
    cat_map = {cat["id"]: idx for idx, cat in enumerate(coco["categories"])}

    # Group annotations by image
    from collections import defaultdict
    ann_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        ann_by_image[ann["image_id"]].append(ann)

    for img_id, anns in ann_by_image.items():
        img_info = id_to_image[img_id]
        W, H = img_info["width"], img_info["height"]
        stem = Path(img_info["file_name"]).stem

        lines = []
        for ann in anns:
            cls_id = cat_map[ann["category_id"]]
            x, y, w, h = ann["bbox"]  # COCO: top-left x,y,w,h
            cx = (x + w / 2) / W
            cy = (y + h / 2) / H
            nw = w / W
            nh = h / H
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        label_file = output_path / f"{stem}.txt"
        label_file.write_text("\n".join(lines))

    logger.info(f"Converted {len(ann_by_image)} annotations → {output_dir}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv8 for document region detection")
    p.add_argument("--data", default="data.yaml", help="Dataset YAML path")
    p.add_argument("--model", default="yolov8n.pt", help="Base model (n/s/m/l/x)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="cpu", help="cpu / 0 / 0,1")
    p.add_argument("--resume", default=None, help="Resume from checkpoint .pt path")
    p.add_argument("--export", default=None, help="Export format after training (onnx/torchscript)")
    p.add_argument("--validate-only", action="store_true", help="Only run val, no training")
    p.add_argument("--weights", default=None, help="Weights for validation/export")
    p.add_argument(
        "--convert-coco",
        nargs=3,
        metavar=("COCO_JSON", "OUTPUT_DIR", "IMAGE_DIR"),
        help="Convert COCO annotations to YOLO format"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.convert_coco:
        convert_coco_to_yolo(*args.convert_coco)
    elif args.validate_only:
        trainer = DocumentYOLOTrainer(device=args.device)
        trainer.validate(args.data, args.weights or "best.pt")
    else:
        model_path = args.resume or args.model
        trainer = DocumentYOLOTrainer(model_path=model_path, device=args.device)
        results = trainer.train(
            data_yaml=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
        )
        if args.export:
            best = Path(results.save_dir) / "weights" / "best.pt"
            trainer.export(str(best), format=args.export)
