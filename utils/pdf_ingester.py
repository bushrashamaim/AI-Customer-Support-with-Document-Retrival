"""
PDF Ingestion Utility
Converts PDF pages to numpy arrays for the CV pipeline.
Uses PyMuPDF (fitz) for fast, accurate rendering.
"""

import numpy as np
from pathlib import Path
from loguru import logger
from typing import Union, Generator


class PDFIngester:
    """
    Loads PDF files and renders each page as a high-resolution numpy array.
    """

    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        self._zoom = dpi / 72.0  # fitz uses 72dpi as base

    def load(self, source: Union[str, Path, bytes]) -> "PDFDocument":
        """Open a PDF from path or bytes."""
        import fitz
        if isinstance(source, (str, Path)):
            doc = fitz.open(str(source))
            logger.info(f"Opened PDF: {source} ({doc.page_count} pages)")
        elif isinstance(source, bytes):
            doc = fitz.open(stream=source, filetype="pdf")
            logger.info(f"Opened PDF from bytes ({doc.page_count} pages)")
        else:
            raise TypeError(f"Unsupported source: {type(source)}")
        return PDFDocument(doc, dpi=self.dpi, zoom=self._zoom)


class PDFDocument:
    """Wrapper around a fitz PDF document for iteration and metadata."""

    def __init__(self, doc, dpi: int, zoom: float):
        self._doc = doc
        self.dpi = dpi
        self.zoom = zoom
        self.page_count = doc.page_count

    @property
    def metadata(self) -> dict:
        meta = self._doc.metadata
        return {
            "title": meta.get("title", ""),
            "author": meta.get("author", ""),
            "subject": meta.get("subject", ""),
            "creator": meta.get("creator", ""),
            "page_count": self.page_count,
        }

    def render_page(self, page_index: int) -> np.ndarray:
        """Render a single page to a numpy BGR array."""
        import fitz
        page = self._doc.load_page(page_index)
        matrix = fitz.Matrix(self.zoom, self.zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        import cv2
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def pages(self, page_range: slice = None) -> Generator[tuple[int, np.ndarray], None, None]:
        """
        Iterate over rendered pages.
        Yields (page_number_1indexed, image_array) tuples.
        """
        indices = range(self.page_count)
        if page_range:
            indices = range(*page_range.indices(self.page_count))

        for i in indices:
            logger.debug(f"Rendering page {i + 1}/{self.page_count}")
            yield i + 1, self.render_page(i)

    def extract_embedded_text(self, page_index: int) -> str:
        """
        Extract embedded text directly from PDF (no OCR needed for digital PDFs).
        Returns empty string for scanned documents.
        """
        page = self._doc.load_page(page_index)
        return page.get_text()

    def is_scanned(self, sample_pages: int = 3) -> bool:
        """
        Heuristic: if embedded text is very sparse, document is likely scanned.
        """
        checked = min(sample_pages, self.page_count)
        total_chars = sum(
            len(self.extract_embedded_text(i))
            for i in range(checked)
        )
        avg = total_chars / checked
        is_scan = avg < 50
        logger.info(f"Scanned document heuristic: avg chars/page={avg:.0f} → {is_scan}")
        return is_scan

    def close(self):
        self._doc.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
