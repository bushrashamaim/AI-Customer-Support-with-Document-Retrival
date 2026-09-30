"""
Unit Tests — Document CV Pipeline
Run with: pytest tests/test_pipeline.py -v
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Preprocessor Tests ───────────────────────────────────────────────────────

class TestDocumentPreprocessor:
    def setup_method(self):
        from core.preprocessor import DocumentPreprocessor
        self.pp = DocumentPreprocessor()

    def _make_gray(self, h=200, w=150):
        return np.random.randint(50, 200, (h, w), dtype=np.uint8)

    def _make_bgr(self, h=200, w=150):
        return np.random.randint(50, 200, (h, w, 3), dtype=np.uint8)

    def test_load_ndarray(self):
        img = self._make_bgr()
        loaded = self.pp.load_image(img)
        assert loaded.shape == img.shape

    def test_to_grayscale_from_bgr(self):
        img = self._make_bgr()
        gray = self.pp.to_grayscale(img)
        assert len(gray.shape) == 2

    def test_to_grayscale_passthrough(self):
        gray = self._make_gray()
        result = self.pp.to_grayscale(gray)
        assert result.shape == gray.shape

    def test_enhance_contrast(self):
        gray = self._make_gray()
        enhanced = self.pp.enhance_contrast(gray)
        assert enhanced.shape == gray.shape
        assert enhanced.dtype == np.uint8

    def test_binarize_adaptive(self):
        gray = self._make_gray()
        binary = self.pp.binarize(gray, method="adaptive")
        unique_vals = np.unique(binary)
        assert set(unique_vals).issubset({0, 255})

    def test_binarize_otsu(self):
        gray = self._make_gray()
        binary = self.pp.binarize(gray, method="otsu")
        assert binary.shape == gray.shape

    def test_denoise(self):
        gray = self._make_gray()
        denoised = self.pp.denoise(gray)
        assert denoised.shape == gray.shape

    def test_process_pipeline(self):
        img = self._make_bgr()
        result = self.pp.process(img, deskew=True, denoise=True, binarize=False)
        assert "original" in result
        assert "gray" in result
        assert "processed" in result
        assert result["processed"].shape[:2] == img.shape[:2]

    def test_load_invalid_type(self):
        with pytest.raises(TypeError):
            self.pp.load_image(12345)

    def test_load_invalid_path(self):
        with pytest.raises(FileNotFoundError):
            self.pp.load_image("/nonexistent/path.jpg")


# ─── OCR Result Tests ─────────────────────────────────────────────────────────

class TestOCRDataStructures:
    def test_text_block_properties(self):
        from core.ocr_engine import TextBlock
        bbox = [[10, 20], [100, 20], [100, 50], [10, 50]]
        block = TextBlock(text="Hello World", confidence=0.95, bbox=bbox)
        assert block.x == 10
        assert block.y == 20
        assert block.width == 90
        assert block.height == 30

    def test_text_block_to_dict(self):
        from core.ocr_engine import TextBlock
        bbox = [[0, 0], [50, 0], [50, 20], [0, 20]]
        block = TextBlock(text="Test", confidence=0.8, bbox=bbox)
        d = block.to_dict()
        assert d["text"] == "Test"
        assert "confidence" in d
        assert "bbox" in d
        assert "position" in d

    def test_ocr_result_to_dict(self):
        from core.ocr_engine import OCRResult, TextBlock
        bbox = [[0, 0], [50, 0], [50, 20], [0, 20]]
        result = OCRResult(
            blocks=[TextBlock("Hello", 0.9, bbox)],
            full_text="Hello",
            avg_confidence=0.9,
        )
        d = result.to_dict()
        assert d["full_text"] == "Hello"
        assert d["word_count"] == 1
        assert len(d["blocks"]) == 1


# ─── Detection Tests ──────────────────────────────────────────────────────────

class TestDocumentDetector:
    def setup_method(self):
        from core.detector import DocumentDetector
        self.detector = DocumentDetector(use_fallback=True)

    def _make_doc_image(self, h=800, w=600):
        """Simulate a document with a white background and black text lines."""
        img = np.full((h, w), 240, dtype=np.uint8)
        # Draw a dark horizontal band (simulates a table row)
        img[200:250, 50:550] = 30
        img[300:350, 50:550] = 30
        return img

    def test_detect_returns_result(self):
        img = self._make_doc_image()
        result = self.detector.detect(img)
        assert result is not None
        assert isinstance(result.detections, list)

    def test_detection_result_to_dict(self):
        img = self._make_doc_image()
        result = self.detector.detect(img)
        d = result.to_dict()
        assert "detections" in d
        assert "total_detections" in d
        assert "label_counts" in d

    def test_detection_crop(self):
        from core.detector import Detection
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        det = Detection(label="table", confidence=0.8, x=10, y=10, width=50, height=30)
        crop = det.crop(img)
        assert crop.shape == (30, 50, 3)

    def test_detection_xyxy(self):
        from core.detector import Detection
        det = Detection(label="logo", confidence=0.7, x=5, y=10, width=40, height=20)
        assert det.xyxy == (5, 10, 45, 30)

    def test_visualize(self):
        img = self._make_doc_image()
        result = self.detector.detect(img)
        vis = self.detector.visualize(img, result)
        assert vis.shape[2] == 3  # Should be BGR


# ─── Classifier Tests ─────────────────────────────────────────────────────────

class TestDocumentClassifier:
    def setup_method(self):
        from core.classifier import DocumentClassifier
        self.clf = DocumentClassifier()

    def test_classify_invoice(self):
        text = "Invoice #12345\nBill To: John Doe\nSubtotal: $500\nTax: $50\nAmount Due: $550"
        result = self.clf.classify(text)
        assert result.predicted_type == "invoice"
        assert result.confidence > 0

    def test_classify_contract(self):
        text = "This Agreement is made between the parties hereinafter referred to. The obligations and liability shall be governed by law."
        result = self.clf.classify(text)
        assert result.predicted_type == "contract"

    def test_classify_resume(self):
        text = "Curriculum Vitae\nWork Experience\nEducation\nSkills\nReferences available upon request"
        result = self.clf.classify(text)
        assert result.predicted_type == "resume"

    def test_classify_unknown(self):
        text = ""
        result = self.clf.classify(text)
        assert result.predicted_type == "unknown"

    def test_classify_returns_scores(self):
        text = "Receipt\nThank you\nTotal: $20"
        result = self.clf.classify(text)
        assert isinstance(result.scores, dict)
        assert len(result.scores) > 0

    def test_extract_entities_emails(self):
        text = "Contact us at support@example.com or sales@company.org"
        entities = self.clf.extract_entities(text)
        assert "emails" in entities
        assert len(entities["emails"]) == 2

    def test_extract_entities_dates(self):
        text = "Invoice date: 01/15/2024. Due: 02/15/2024"
        entities = self.clf.extract_entities(text)
        assert "dates" in entities
        assert len(entities["dates"]) >= 2

    def test_extract_entities_amounts(self):
        text = "Total amount: $1,250.00"
        entities = self.clf.extract_entities(text)
        assert "amounts" in entities

    def test_classification_result_to_dict(self):
        text = "Invoice #001\nAmount Due: $100"
        result = self.clf.classify(text)
        d = result.to_dict()
        assert "predicted_type" in d
        assert "confidence" in d
        assert "top_3" in d
        assert len(d["top_3"]) <= 3


# ─── Pipeline Integration Test ────────────────────────────────────────────────

class TestDocumentPipeline:
    def setup_method(self):
        from core.pipeline import DocumentPipeline
        self.pipeline = DocumentPipeline(ocr_languages=["en"], use_gpu=False)

    def _make_test_image(self):
        """Create a synthetic document image."""
        import cv2
        img = np.full((800, 600, 3), 240, dtype=np.uint8)
        cv2.putText(img, "INVOICE", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        cv2.putText(img, "Amount Due: $500", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(img, "Bill To: John Doe", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
        return img

    def test_process_image_array(self):
        img = self._make_test_image()
        result = self.pipeline.process_image(img, file_name="test_invoice.png")
        assert result.page_count == 1
        assert len(result.pages) == 1
        assert result.total_time_ms > 0
        assert result.document_type in [
            "invoice", "receipt", "form", "letter", "unknown",
            "contract", "id_card", "passport", "bank_statement",
            "resume", "medical_report", "legal_document",
        ]

    def test_result_summary(self):
        img = self._make_test_image()
        result = self.pipeline.process_image(img)
        summary = result.summary()
        assert "document_type" in summary
        assert "page_count" in summary
        assert "total_word_count" in summary

    def test_result_to_dict(self):
        img = self._make_test_image()
        result = self.pipeline.process_image(img)
        d = result.to_dict()
        assert "pages" in d
        assert len(d["pages"]) == 1
        page = d["pages"][0]
        assert "ocr" in page
        assert "detections" in page
        assert "classification" in page
        assert "entities" in page
