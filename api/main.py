"""
Document CV API — FastAPI
Endpoints:
  POST /analyze          — upload image or PDF, get full analysis
  POST /analyze/summary  — lightweight summary only
  GET  /health           — service health check
"""

import io
import time
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.pipeline import DocumentPipeline


# ─── Lifespan: load pipeline once at startup ────────────────────────────────

pipeline: Optional[DocumentPipeline] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    logger.info("Starting up — loading DocumentPipeline...")
    pipeline = DocumentPipeline(
        ocr_languages=["en"],
        use_gpu=False,
        min_ocr_confidence=0.3,
    )
    logger.info("Pipeline ready.")
    yield
    logger.info("Shutting down.")


# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Document CV API",
    description="AI-powered document classification, OCR, and region detection.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_TYPES = {
    "application/pdf",
    "image/jpeg", "image/jpg", "image/png",
    "image/tiff", "image/bmp", "image/webp",
}
MAX_FILE_SIZE_MB = 20


# ─── Response schemas ─────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    pipeline_loaded: bool
    timestamp: float


class AnalysisSummary(BaseModel):
    file_name: str
    page_count: int
    document_type: str
    confidence: float
    total_word_count: int
    total_time_ms: float


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _validate_upload(file: UploadFile) -> None:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. "
                   f"Allowed: {', '.join(sorted(ALLOWED_TYPES))}",
        )


async def _read_bytes(file: UploadFile) -> bytes:
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Max: {MAX_FILE_SIZE_MB} MB",
        )
    return content


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    return HealthResponse(
        status="ok",
        pipeline_loaded=pipeline is not None,
        timestamp=time.time(),
    )


@app.post("/analyze", tags=["Analysis"])
async def analyze_document(
    file: UploadFile = File(..., description="PDF or image file to analyze"),
    max_pages: Optional[int] = Query(None, ge=1, le=50, description="Max PDF pages to process"),
    summary_only: bool = Query(False, description="Return summary only (faster)"),
):
    """
    Full document analysis: OCR + region detection + classification + entity extraction.

    Returns detailed per-page breakdown or a lightweight summary.
    """
    _validate_upload(file)
    content = await _read_bytes(file)

    try:
        result = pipeline.process(
            content,
            file_name=file.filename,
            max_pages=max_pages,
        )
    except Exception as e:
        logger.exception(f"Pipeline error on {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

    if summary_only:
        return JSONResponse(content=result.summary())

    return JSONResponse(content=result.to_dict())


@app.post("/analyze/summary", response_model=AnalysisSummary, tags=["Analysis"])
async def analyze_summary(
    file: UploadFile = File(...),
    max_pages: Optional[int] = Query(None, ge=1, le=50),
):
    """Lightweight analysis — returns document type and word count only."""
    _validate_upload(file)
    content = await _read_bytes(file)

    try:
        result = pipeline.process(content, file_name=file.filename, max_pages=max_pages)
    except Exception as e:
        logger.exception(f"Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return AnalysisSummary(**result.summary())


@app.post("/ocr", tags=["Analysis"])
async def ocr_only(
    file: UploadFile = File(...),
    max_pages: Optional[int] = Query(None, ge=1, le=50),
):
    """Extract text only (OCR). Faster than full analysis."""
    _validate_upload(file)
    content = await _read_bytes(file)

    try:
        result = pipeline.process(content, file_name=file.filename, max_pages=max_pages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse(content={
        "file_name": result.file_name,
        "page_count": result.page_count,
        "full_text": result.total_text,
        "pages": [
            {
                "page": p.page_number,
                "text": p.ocr.full_text,
                "word_count": len(p.ocr.full_text.split()),
                "avg_confidence": p.ocr.avg_confidence,
            }
            for p in result.pages
        ],
    })


@app.post("/detect", tags=["Analysis"])
async def detect_only(
    file: UploadFile = File(...),
):
    """Detect document regions only (tables, signatures, logos, etc.)."""
    if file.content_type not in ALLOWED_TYPES - {"application/pdf"}:
        raise HTTPException(status_code=415, detail="Images only for /detect endpoint.")

    content = await _read_bytes(file)

    try:
        result = pipeline.process_image(content, file_name=file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    page = result.pages[0]
    return JSONResponse(content={
        "file_name": result.file_name,
        "detections": page.detections.to_dict(),
    })
