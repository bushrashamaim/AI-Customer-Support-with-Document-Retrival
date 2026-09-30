"""
OCR Module — Text Extraction via EasyOCR
Supports multi-language recognition, confidence filtering,
and structured text block output.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger


@dataclass
class TextBlock:
    """A single recognized text region."""
    text: str
    confidence: float
    bbox: list  # [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
    line_number: int = 0

    @property
    def x(self) -> int:
        return int(self.bbox[0][0])

    @property
    def y(self) -> int:
        return int(self.bbox[0][1])

    @property
    def width(self) -> int:
        return int(self.bbox[2][0]) - int(self.bbox[0][0])

    @property
    def height(self) -> int:
        return int(self.bbox[2][1]) - int(self.bbox[0][1])

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "bbox": self.bbox,
            "position": {"x": self.x, "y": self.y, "w": self.width, "h": self.height},
            "line_number": self.line_number,
        }


@dataclass
class OCRResult:
    """Full OCR result for a document page."""
    blocks: list[TextBlock] = field(default_factory=list)
    full_text: str = ""
    language: str = "en"
    avg_confidence: float = 0.0
    page_number: int = 1

    def to_dict(self) -> dict:
        return {
            "page_number": self.page_number,
            "language": self.language,
            "full_text": self.full_text,
            "avg_confidence": round(self.avg_confidence, 4),
            "blocks": [b.to_dict() for b in self.blocks],
            "word_count": len(self.full_text.split()),
        }


class OCREngine:
    """
    EasyOCR-based text extraction engine.
    Lazy-loads the model on first use to avoid startup delays.
    """

    def __init__(self, languages: list[str] = None, gpu: bool = False):
        self.languages = languages or ["en"]
        self.gpu = gpu
        self._reader = None
        logger.info(f"OCREngine initialized (langs={self.languages}, gpu={gpu})")

    @property
    def reader(self):
        """Lazy-load EasyOCR reader."""
        if self._reader is None:
            import easyocr
            logger.info("Loading EasyOCR model (first use, may take a moment)...")
            self._reader = easyocr.Reader(self.languages, gpu=self.gpu)
            logger.info("EasyOCR model loaded.")
        return self._reader

    def warmup(self):
        """Pre-load the EasyOCR model so first document isn't slow."""
        _ = self.reader
        logger.info("OCR model warmed up and ready.")

    def extract(
        self,
        image: np.ndarray,
        min_confidence: float = 0.3,
        paragraph: bool = False,
        page_number: int = 1,
    ) -> OCRResult:
        """
        Run OCR on a preprocessed image.

        Args:
            image:          Grayscale or BGR numpy array
            min_confidence: Filter out blocks below this threshold
            paragraph:      Group text into paragraphs
            page_number:    Page index for multi-page docs

        Returns:
            OCRResult with all text blocks
        """
        logger.info(f"Running OCR on page {page_number}...")
        raw = self.reader.readtext(image, paragraph=paragraph, detail=1)

        blocks = []
        for idx, (bbox, text, conf) in enumerate(raw):
            if conf < min_confidence or not text.strip():
                continue
            blocks.append(TextBlock(
                text=text.strip(),
                confidence=conf,
                bbox=bbox,
                line_number=idx,
            ))

        blocks.sort(key=lambda b: b.y)

        full_text = "\n".join(b.text for b in blocks)
        avg_conf = float(np.mean([b.confidence for b in blocks])) if blocks else 0.0

        result = OCRResult(
            blocks=blocks,
            full_text=full_text,
            language=self.languages[0],
            avg_confidence=avg_conf,
            page_number=page_number,
        )
        logger.info(
            f"OCR page {page_number}: {len(blocks)} blocks, "
            f"{len(full_text.split())} words, avg_conf={avg_conf:.2f}"
        )
        return result

    def extract_from_regions(
        self,
        image: np.ndarray,
        regions: list[dict],
        min_confidence: float = 0.3,
    ) -> dict[str, OCRResult]:
        """
        Run OCR on specific detected regions (e.g., from YOLO output).

        Args:
            image:   Full document image
            regions: List of region dicts with 'label', 'x', 'y', 'w', 'h'

        Returns:
            Dict mapping region label → OCRResult
        """
        results = {}
        h, w = image.shape[:2]

        for region in regions:
            x, y = max(0, region["x"]), max(0, region["y"])
            rw, rh = region["w"], region["h"]
            x2, y2 = min(w, x + rw), min(h, y + rh)

            crop = image[y:y2, x:x2]
            label = region.get("label", "region")
            logger.debug(f"OCR on region '{label}' [{x},{y},{rw},{rh}]")

            results[label] = self.extract(crop, min_confidence=min_confidence)

        return results