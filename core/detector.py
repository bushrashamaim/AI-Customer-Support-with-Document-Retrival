"""
Document Region Detector — YOLOv8-based
Detects semantic regions: tables, signatures, stamps, logos,
headers, footers, checkboxes, and form fields.
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from pathlib import Path
from loguru import logger
from typing import Optional


# Default label set for document layout detection
DOCUMENT_LABELS = [
    "text_block", "table", "figure", "signature",
    "stamp", "logo", "header", "footer",
    "checkbox", "form_field", "barcode", "qrcode",
]


@dataclass
class Detection:
    """Single detected region."""
    label: str
    confidence: float
    x: int
    y: int
    width: int
    height: int
    class_id: int = 0

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "bbox": {"x": self.x, "y": self.y, "w": self.width, "h": self.height},
            "area": self.width * self.height,
        }

    def crop(self, image: np.ndarray) -> np.ndarray:
        """Return the cropped region from a full image."""
        h, w = image.shape[:2]
        x1, y1, x2, y2 = self.xyxy
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        return image[y1:y2, x1:x2]


@dataclass
class DetectionResult:
    """All detections for a single document page."""
    detections: list[Detection] = field(default_factory=list)
    page_number: int = 1
    image_shape: tuple = (0, 0)

    def filter_by_label(self, label: str) -> list[Detection]:
        return [d for d in self.detections if d.label == label]

    def filter_by_confidence(self, min_conf: float) -> list[Detection]:
        return [d for d in self.detections if d.confidence >= min_conf]

    def has(self, label: str) -> bool:
        return any(d.label == label for d in self.detections)

    def to_dict(self) -> dict:
        labels = {}
        for d in self.detections:
            labels.setdefault(d.label, 0)
            labels[d.label] += 1
        return {
            "page_number": self.page_number,
            "total_detections": len(self.detections),
            "label_counts": labels,
            "detections": [d.to_dict() for d in self.detections],
        }


class DocumentDetector:
    """
    YOLOv8-based document region detector.
    Falls back to rule-based heuristics if model weights aren't available.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.4,
        iou_threshold: float = 0.45,
        labels: list[str] = None,
        use_fallback: bool = True,
    ):
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.labels = labels or DOCUMENT_LABELS
        self.use_fallback = use_fallback
        self._model = None
        self._model_path = model_path
        self._model_available = False

        if model_path:
            self._load_model(model_path)

    def _load_model(self, model_path: str):
        try:
            from ultralytics import YOLO
            self._model = YOLO(model_path)
            self._model_available = True
            logger.info(f"YOLO model loaded from: {model_path}")
        except Exception as e:
            logger.warning(f"Failed to load YOLO model: {e}. Will use heuristic fallback.")

    def detect(
        self,
        image: np.ndarray,
        page_number: int = 1,
    ) -> DetectionResult:
        """
        Detect document regions in image.
        Uses YOLO if model available, else heuristic fallback.
        """
        h, w = image.shape[:2]
        result = DetectionResult(page_number=page_number, image_shape=(h, w))

        if self._model_available:
            result.detections = self._yolo_detect(image)
        elif self.use_fallback:
            logger.debug("Using heuristic region detection (no YOLO model).")
            result.detections = self._heuristic_detect(image)

        logger.info(
            f"Detection page {page_number}: {len(result.detections)} regions found — "
            + str({k: v for k, v in result.to_dict()["label_counts"].items()})
        )
        return result

    def _yolo_detect(self, image: np.ndarray) -> list[Detection]:
        """Run YOLOv8 inference."""
        results = self._model(
            image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = self.labels[cls_id] if cls_id < len(self.labels) else f"class_{cls_id}"
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append(Detection(
                    label=label,
                    confidence=float(box.conf[0]),
                    x=x1, y=y1,
                    width=x2 - x1,
                    height=y2 - y1,
                    class_id=cls_id,
                ))
        return detections

    def _heuristic_detect(self, image: np.ndarray) -> list[Detection]:
        """
        Rule-based fallback: detect tables via contours,
        headers/footers by position, and logos by top-left blob.
        """
        detections = []
        h, w = image.shape[:2]

        gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1000:
                continue
            rx, ry, rw, rh = cv2.boundingRect(cnt)
            aspect = rw / (rh + 1e-6)

            # Heuristic: wide + tall → likely a table
            if rw > w * 0.5 and rh > h * 0.1 and 1.5 < aspect < 10:
                detections.append(Detection(
                    label="table", confidence=0.6,
                    x=rx, y=ry, width=rw, height=rh,
                ))
            # Heuristic: top 10% of page → header
            elif ry < h * 0.1 and rw > w * 0.3:
                detections.append(Detection(
                    label="header", confidence=0.55,
                    x=rx, y=ry, width=rw, height=rh,
                ))
            # Heuristic: bottom 10% → footer
            elif ry + rh > h * 0.9 and rw > w * 0.3:
                detections.append(Detection(
                    label="footer", confidence=0.55,
                    x=rx, y=ry, width=rw, height=rh,
                ))

        return detections

    def visualize(
        self,
        image: np.ndarray,
        result: DetectionResult,
        color_map: dict = None,
    ) -> np.ndarray:
        """Draw bounding boxes with labels on image."""
        vis = image.copy()
        if len(vis.shape) == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

        default_colors = {
            "table": (0, 200, 100),
            "header": (200, 100, 0),
            "footer": (200, 100, 0),
            "signature": (0, 100, 220),
            "stamp": (220, 50, 50),
            "logo": (180, 0, 220),
            "figure": (0, 180, 220),
        }
        colors = {**default_colors, **(color_map or {})}

        for det in result.detections:
            color = colors.get(det.label, (150, 150, 150))
            x1, y1, x2, y2 = det.xyxy
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label_text = f"{det.label} {det.confidence:.2f}"
            cv2.putText(
                vis, label_text, (x1, max(y1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
            )

        return vis
