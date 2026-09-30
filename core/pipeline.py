"""
Document CV Pipeline — Main Orchestrator
Ties together: ingestion → preprocessing → OCR → detection → classification
"""

import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Union, Optional
from loguru import logger

from core.preprocessor import DocumentPreprocessor
from core.ocr_engine import OCREngine, OCRResult
from core.detector import DocumentDetector, DetectionResult
from core.classifier import DocumentClassifier, ClassificationResult


@dataclass
class PageResult:
    """Full analysis result for a single page."""
    page_number: int
    ocr: OCRResult
    detections: DetectionResult
    classification: ClassificationResult
    entities: dict
    processing_time_ms: float

    def to_dict(self) -> dict:
        return {
            "page_number": self.page_number,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "classification": self.classification.to_dict(),
            "ocr": self.ocr.to_dict(),
            "detections": self.detections.to_dict(),
            "entities": self.entities,
        }


@dataclass
class DocumentResult:
    """Full analysis result for an entire document."""
    file_name: str
    file_type: str  # 'pdf' | 'image'
    page_count: int
    pages: list[PageResult] = field(default_factory=list)
    document_type: str = "unknown"
    document_confidence: float = 0.0
    total_text: str = ""
    metadata: dict = field(default_factory=dict)
    total_time_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "file_name": self.file_name,
            "file_type": self.file_type,
            "page_count": self.page_count,
            "document_type": self.document_type,
            "document_confidence": round(self.document_confidence, 4),
            "total_time_ms": round(self.total_time_ms, 2),
            "total_word_count": len(self.total_text.split()),
            "metadata": self.metadata,
            "pages": [p.to_dict() for p in self.pages],
        }

    def summary(self) -> dict:
        """Lightweight summary without per-page OCR blocks."""
        return {
            "file_name": self.file_name,
            "page_count": self.page_count,
            "document_type": self.document_type,
            "confidence": round(self.document_confidence, 4),
            "total_word_count": len(self.total_text.split()),
            "total_time_ms": round(self.total_time_ms, 2),
        }


class DocumentPipeline:
    """
    End-to-end document analysis pipeline.

    Usage:
        pipeline = DocumentPipeline()
        result = pipeline.process("invoice.pdf")
        print(result.document_type)
    """

    def __init__(
        self,
        ocr_languages: list[str] = None,
        yolo_model_path: Optional[str] = None,
        ml_classifier_path: Optional[str] = None,
        use_gpu: bool = False,
        min_ocr_confidence: float = 0.3,
        dpi: int = 150,  # was 300 — reduced for faster processing
    ):
        logger.info("Initializing DocumentPipeline...")
        self.preprocessor = DocumentPreprocessor(target_dpi=dpi)
        self.ocr = OCREngine(languages=ocr_languages or ["en"], gpu=use_gpu)
        self.detector = DocumentDetector(
            model_path=yolo_model_path,
            confidence_threshold=0.4,
            use_fallback=True,
        )
        self.classifier = DocumentClassifier(ml_model_path=ml_classifier_path)
        self.min_ocr_confidence = min_ocr_confidence
        self.dpi = dpi
        # Warmup: load OCR model now so first document doesn't hang
        logger.info("Warming up OCR model...")
        self.ocr.warmup()
        logger.info("Pipeline ready.")

    def process_image(
        self,
        source: Union[str, Path, np.ndarray, bytes],
        file_name: str = "document",
    ) -> DocumentResult:
        """Process a single image file."""
        t0 = time.perf_counter()

        prep = self.preprocessor.process(source)
        page_result = self._analyze_page(prep["processed"], prep["original"], page_number=1)

        total_ms = (time.perf_counter() - t0) * 1000
        result = DocumentResult(
            file_name=file_name,
            file_type="image",
            page_count=1,
            pages=[page_result],
            document_type=page_result.classification.predicted_type,
            document_confidence=page_result.classification.confidence,
            total_text=page_result.ocr.full_text,
            total_time_ms=total_ms,
        )
        logger.success(
            f"[{file_name}] → {result.document_type} "
            f"(conf={result.document_confidence:.2f}, {total_ms:.0f}ms)"
        )
        return result

    def process_pdf(
        self,
        source: Union[str, Path, bytes],
        file_name: str = "document.pdf",
        max_pages: Optional[int] = None,
    ) -> DocumentResult:
        """Process a PDF document (all pages or up to max_pages)."""
        from utils.pdf_ingester import PDFIngester
        t0 = time.perf_counter()

        ingester = PDFIngester(dpi=self.dpi)
        with ingester.load(source) as pdf_doc:
            metadata = pdf_doc.metadata
            is_scanned = pdf_doc.is_scanned()
            pages_results = []
            all_text_parts = []

            for page_num, page_img in pdf_doc.pages():
                if max_pages and page_num > max_pages:
                    break

                embedded = "" if is_scanned else pdf_doc.extract_embedded_text(page_num - 1)

                prep = self.preprocessor.process(page_img)
                page_result = self._analyze_page(
                    prep["processed"], prep["original"],
                    page_number=page_num,
                    embedded_text=embedded,
                )
                pages_results.append(page_result)
                all_text_parts.append(page_result.ocr.full_text)

        total_text = "\n\n".join(all_text_parts)
        doc_cls = self.classifier.classify(total_text)

        total_ms = (time.perf_counter() - t0) * 1000
        result = DocumentResult(
            file_name=file_name,
            file_type="pdf",
            page_count=len(pages_results),
            pages=pages_results,
            document_type=doc_cls.predicted_type,
            document_confidence=doc_cls.confidence,
            total_text=total_text,
            metadata=metadata,
            total_time_ms=total_ms,
        )
        logger.success(
            f"[{file_name}] {len(pages_results)} pages → {result.document_type} "
            f"(conf={result.document_confidence:.2f}, {total_ms:.0f}ms)"
        )
        return result

    def process(
        self,
        source: Union[str, Path, bytes],
        file_name: Optional[str] = None,
        max_pages: Optional[int] = None,
    ) -> DocumentResult:
        """
        Auto-detect file type and run appropriate pipeline.
        Accepts file paths, bytes, or numpy arrays.
        """
        if isinstance(source, (str, Path)):
            p = Path(source)
            fname = file_name or p.name
            if p.suffix.lower() == ".pdf":
                return self.process_pdf(source, file_name=fname, max_pages=max_pages)
            else:
                return self.process_image(source, file_name=fname)
        elif isinstance(source, bytes):
            if source[:4] == b"%PDF":
                return self.process_pdf(source, file_name=file_name or "upload.pdf", max_pages=max_pages)
            else:
                return self.process_image(source, file_name=file_name or "upload")
        elif isinstance(source, np.ndarray):
            return self.process_image(source, file_name=file_name or "image")
        else:
            raise TypeError(f"Unsupported input type: {type(source)}")

    def _analyze_page(
        self,
        processed_img: np.ndarray,
        original_img: np.ndarray,
        page_number: int = 1,
        embedded_text: str = "",
    ) -> PageResult:
        """Run OCR + detection + classification on a single preprocessed page."""
        t0 = time.perf_counter()

        ocr_result = self.ocr.extract(processed_img, min_confidence=self.min_ocr_confidence, page_number=page_number)
        if embedded_text:
            ocr_result.full_text = embedded_text + "\n" + ocr_result.full_text

        det_result = self.detector.detect(original_img, page_number=page_number)

        cls_result = self.classifier.classify(
            ocr_result.full_text,
            detection_result=det_result,
            image_shape=original_img.shape,
        )

        entities = self.classifier.extract_entities(ocr_result.full_text)

        page_ms = (time.perf_counter() - t0) * 1000

        return PageResult(
            page_number=page_number,
            ocr=ocr_result,
            detections=det_result,
            classification=cls_result,
            entities=entities,
            processing_time_ms=page_ms,
        )