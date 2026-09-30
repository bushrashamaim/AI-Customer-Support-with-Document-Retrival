"""
Document Classifier
Classifies documents by type (invoice, contract, ID card, receipt, etc.)
using a combination of OCR text features and layout features.
"""

import re
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger


# Document type taxonomy
DOCUMENT_TYPES = [
    "invoice",
    "contract",
    "id_card",
    "passport",
    "receipt",
    "bank_statement",
    "resume",
    "medical_report",
    "legal_document",
    "form",
    "letter",
    "unknown",
]


# Keyword signals per document type
KEYWORD_SIGNALS: dict[str, list[str]] = {
    "invoice": [
        "invoice", "bill to", "amount due", "subtotal", "tax", "vat",
        "invoice #", "invoice no", "payment due", "total amount",
    ],
    "contract": [
        "agreement", "whereas", "hereinafter", "party", "parties",
        "terms and conditions", "obligations", "liability", "indemnify",
        "governing law", "jurisdiction", "signed by",
    ],
    "id_card": [
        "national id", "identity card", "date of birth", "dob",
        "nationality", "id number", "expiry", "issued by",
    ],
    "passport": [
        "passport", "nationality", "place of birth", "date of issue",
        "date of expiry", "mrz", "surname", "given names",
    ],
    "receipt": [
        "receipt", "thank you", "change", "cash", "subtotal", "total",
        "qty", "item", "cashier", "transaction", "payment method",
    ],
    "bank_statement": [
        "account number", "statement", "balance", "debit", "credit",
        "transaction date", "opening balance", "closing balance", "iban",
    ],
    "resume": [
        "curriculum vitae", "cv", "resume", "work experience",
        "education", "skills", "objective", "references", "employment",
    ],
    "medical_report": [
        "diagnosis", "patient", "physician", "prescription", "dosage",
        "medical", "symptoms", "treatment", "blood pressure", "lab results",
    ],
    "legal_document": [
        "plaintiff", "defendant", "court", "hereby", "affidavit",
        "sworn", "notary", "judgement", "pursuant", "ordinance",
    ],
    "form": [
        "please fill", "check all that apply", "signature", "date",
        "first name", "last name", "address", "phone number",
    ],
    "letter": [
        "dear", "sincerely", "regards", "to whom it may concern",
        "yours faithfully", "re:", "subject:",
    ],
}


@dataclass
class ClassificationResult:
    """Result of document classification."""
    predicted_type: str
    confidence: float
    scores: dict[str, float] = field(default_factory=dict)
    detected_keywords: dict[str, list[str]] = field(default_factory=dict)
    method: str = "keyword"

    def to_dict(self) -> dict:
        return {
            "predicted_type": self.predicted_type,
            "confidence": round(self.confidence, 4),
            "top_3": sorted(self.scores.items(), key=lambda x: x[1], reverse=True)[:3],
            "detected_keywords": {k: v for k, v in self.detected_keywords.items() if v},
            "method": self.method,
        }


class DocumentClassifier:
    """
    Multi-strategy document classifier.

    Strategy 1: Keyword matching (fast, interpretable)
    Strategy 2: Layout features (table count, image count, aspect ratio)
    Strategy 3: ML model (sklearn TF-IDF + SVM — requires training data)
    """

    def __init__(self, ml_model_path: Optional[str] = None):
        self._ml_model = None
        self._vectorizer = None
        if ml_model_path:
            self._load_ml_model(ml_model_path)

    def _load_ml_model(self, path: str):
        try:
            import joblib
            data = joblib.load(path)
            self._ml_model = data["model"]
            self._vectorizer = data["vectorizer"]
            logger.info(f"ML classifier loaded from {path}")
        except Exception as e:
            logger.warning(f"Could not load ML model: {e}")

    def classify_by_keywords(self, text: str) -> ClassificationResult:
        """Score document type by keyword frequency in OCR text."""
        text_lower = text.lower()
        scores = {}
        detected = {}

        for doc_type, keywords in KEYWORD_SIGNALS.items():
            hits = [kw for kw in keywords if kw in text_lower]
            score = len(hits) / len(keywords) if keywords else 0
            # Boost for multiple hits
            if len(hits) > 3:
                score = min(score * 1.5, 1.0)
            scores[doc_type] = round(score, 4)
            detected[doc_type] = hits

        if not any(scores.values()):
            return ClassificationResult(
                predicted_type="unknown",
                confidence=0.0,
                scores=scores,
                detected_keywords=detected,
                method="keyword",
            )

        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]

        return ClassificationResult(
            predicted_type=best_type,
            confidence=best_score,
            scores=scores,
            detected_keywords=detected,
            method="keyword",
        )

    def classify_by_layout(
        self,
        detection_result,
        image_shape: tuple,
    ) -> ClassificationResult:
        """
        Use layout signals (table presence, header/footer, aspect ratio)
        to refine or override keyword classification.
        """
        h, w = image_shape[:2]
        aspect = h / (w + 1e-6)
        scores = {t: 0.0 for t in DOCUMENT_TYPES}

        has_table = detection_result.has("table")
        has_signature = detection_result.has("signature")
        has_stamp = detection_result.has("stamp")
        has_logo = detection_result.has("logo")
        table_count = len(detection_result.filter_by_label("table"))

        # Layout heuristics
        if has_table and table_count >= 2:
            scores["invoice"] += 0.4
            scores["bank_statement"] += 0.3

        if has_signature and has_stamp:
            scores["contract"] += 0.5
            scores["legal_document"] += 0.3

        if has_logo and has_table:
            scores["invoice"] += 0.3
            scores["bank_statement"] += 0.2

        # Portrait / near-square → likely ID / card
        if 1.2 < aspect < 1.6 and not has_table:
            scores["id_card"] += 0.3

        # Very tall portrait → resume / letter
        if aspect > 1.3:
            scores["resume"] += 0.15
            scores["letter"] += 0.1

        best = max(scores, key=scores.get)
        return ClassificationResult(
            predicted_type=best if scores[best] > 0 else "unknown",
            confidence=scores[best],
            scores=scores,
            method="layout",
        )

    def classify(
        self,
        ocr_text: str,
        detection_result=None,
        image_shape: tuple = None,
    ) -> ClassificationResult:
        """
        Combined classification using keyword + layout signals.
        ML model used if available.
        """
        # Primary: keyword
        kw_result = self.classify_by_keywords(ocr_text)

        # Secondary: layout (if detection available)
        if detection_result and image_shape:
            layout_result = self.classify_by_layout(detection_result, image_shape)

            # Merge scores
            merged_scores = {}
            for t in DOCUMENT_TYPES:
                kw_score = kw_result.scores.get(t, 0)
                layout_score = layout_result.scores.get(t, 0)
                merged_scores[t] = round(kw_score * 0.7 + layout_score * 0.3, 4)

            best_type = max(merged_scores, key=merged_scores.get)
            best_conf = merged_scores[best_type]

            return ClassificationResult(
                predicted_type=best_type if best_conf > 0.05 else "unknown",
                confidence=best_conf,
                scores=merged_scores,
                detected_keywords=kw_result.detected_keywords,
                method="keyword+layout",
            )

        # ML fallback (if model loaded)
        if self._ml_model and self._vectorizer and ocr_text.strip():
            try:
                vec = self._vectorizer.transform([ocr_text])
                proba = self._ml_model.predict_proba(vec)[0]
                classes = self._ml_model.classes_
                scores = {c: float(p) for c, p in zip(classes, proba)}
                best = max(scores, key=scores.get)
                return ClassificationResult(
                    predicted_type=best,
                    confidence=scores[best],
                    scores=scores,
                    method="ml_model",
                )
            except Exception as e:
                logger.warning(f"ML classification failed: {e}")

        return kw_result

    @staticmethod
    def extract_entities(text: str) -> dict:
        """
        Simple regex-based entity extraction for common document fields.
        Returns dict of entity_type → list of matches.
        """
        patterns = {
            "dates": r"\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})\b",
            "emails": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
            "phone_numbers": r"\+?\d{1,4}[\s\-]?\(?\d{2,4}\)?[\s\-]\d{3,4}[\s\-]?\d{3,4}",
            "amounts": r"(?:Rs\.?|PKR|\$|USD|EUR|GBP)\s*[\d,]+\.?\d{0,2}|\b\d+[\.,]\d{2}\s*(?:USD|EUR|GBP|PKR)\b",
            "invoice_numbers": r"(?:invoice|inv|bill)\s*[#no\.]*\s*([A-Z0-9\-]+)",
            "urls": r"https?://[^\s]+",
        }
        entities = {}
        for name, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                entities[name] = list(set(str(m).strip() for m in matches if m))
        return entities