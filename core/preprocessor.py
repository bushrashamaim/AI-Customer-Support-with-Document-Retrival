"""
Document Preprocessing Pipeline
Handles image enhancement, deskewing, denoising, and binarization
before passing to OCR or detection modules.
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from loguru import logger
from typing import Union


class DocumentPreprocessor:
    """
    Prepares raw document images for downstream CV tasks.
    Applies adaptive thresholding, deskewing, and noise removal.
    """

    def __init__(self, target_dpi: int = 150):
        self.target_dpi = target_dpi

    def load_image(self, source: Union[str, Path, np.ndarray, bytes]) -> np.ndarray:
        """Load image from file path, numpy array, or raw bytes."""
        if isinstance(source, (str, Path)):
            img = cv2.imread(str(source))
            if img is None:
                raise FileNotFoundError(f"Cannot load image: {source}")
            logger.debug(f"Loaded image from {source}: {img.shape}")
            return img
        elif isinstance(source, bytes):
            arr = np.frombuffer(source, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            logger.debug(f"Loaded image from bytes: {img.shape}")
            return img
        elif isinstance(source, np.ndarray):
            return source.copy()
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

    def to_grayscale(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def denoise(self, gray: np.ndarray) -> np.ndarray:
        """Apply Non-Local Means denoising for document clarity."""
        return cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

    def deskew(self, gray: np.ndarray) -> np.ndarray:
        """
        Detect and correct document skew using Hough line transform.
        Returns corrected image.
        """
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

        if lines is None:
            logger.debug("No skew detected, skipping deskew.")
            return gray

        angles = []
        for rho, theta in lines[:, 0]:
            angle = (theta - np.pi / 2) * (180 / np.pi)
            if abs(angle) < 45:
                angles.append(angle)

        if not angles:
            return gray

        median_angle = np.median(angles)
        if abs(median_angle) < 0.5:
            return gray

        logger.debug(f"Correcting skew: {median_angle:.2f}°")
        h, w = gray.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(
            gray, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        return rotated

    def binarize(self, gray: np.ndarray, method: str = "adaptive") -> np.ndarray:
        """
        Binarize image for better OCR results.
        method: 'adaptive' | 'otsu' | 'sauvola'
        """
        if method == "otsu":
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == "adaptive":
            binary = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
        else:
            # Fallback to Otsu
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def enhance_contrast(self, gray: np.ndarray) -> np.ndarray:
        """Apply CLAHE for localized contrast enhancement."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def remove_borders(self, binary: np.ndarray, margin: int = 10) -> np.ndarray:
        """Crop out scanner border artifacts."""
        h, w = binary.shape
        return binary[margin:h - margin, margin:w - margin]

    def process(
        self,
        source: Union[str, Path, np.ndarray, bytes],
        deskew: bool = True,
        denoise: bool = True,
        binarize: bool = False,
        enhance: bool = True,
    ) -> dict:
        """
        Full preprocessing pipeline.

        Returns:
            dict with keys:
                - original: np.ndarray (BGR)
                - gray: np.ndarray
                - processed: np.ndarray (final output)
        """
        original = self.load_image(source)
        gray = self.to_grayscale(original)

        result = gray.copy()

        if enhance:
            result = self.enhance_contrast(result)
        if denoise:
            result = self.denoise(result)
        if deskew:
            result = self.deskew(result)
        if binarize:
            result = self.binarize(result)

        logger.info(f"Preprocessing complete. Output shape: {result.shape}")
        return {
            "original": original,
            "gray": gray,
            "processed": result,
        }
