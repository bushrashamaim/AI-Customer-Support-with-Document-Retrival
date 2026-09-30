# 📄 Document CV Pipeline

> AI-powered document analysis: **OCR + Object Detection + Classification + Entity Extraction**

---

## 🗂️ Full Project Structure

```
doc_vision/
│
├── core/                        ← Core CV modules
│   ├── preprocessor.py          OCR-prep: deskew, denoise, CLAHE
│   ├── ocr_engine.py            EasyOCR wrapper with TextBlock output
│   ├── detector.py              YOLOv8 region detector + heuristic fallback
│   ├── classifier.py            Document type classifier + entity extractor
│   └── pipeline.py              Main orchestrator (image/PDF → results)
│
├── api/
│   └── main.py                  FastAPI REST API (5 endpoints)
│
├── ui/
│   └── dashboard.py             Streamlit web dashboard
│
├── training/
│   ├── train_yolo.py            YOLOv8 fine-tuning + COCO→YOLO converter
│   └── train_classifier.py      TF-IDF + SVM/LR/RF training + GridSearch
│
├── database/
│   └── db_manager.py            SQLite: store, search, analytics
│
├── scripts/
│   └── batch_process.py         Bulk folder processing with CSV/JSON export
│
├── utils/
│   └── pdf_ingester.py          PyMuPDF PDF → numpy arrays
│
├── tests/
│   └── test_pipeline.py         25+ pytest unit tests
│
├── Dockerfile                   API container
├── Dockerfile.ui                Streamlit container
├── docker-compose.yml           Full stack (API + UI + Batch)
├── .env.example                 Environment config template
├── requirements.txt
└── run.py                       CLI entry point
```

---

## ⚡ Quick Start

### Option 1 — Docker (Recommended)

```bash
git clone <repo> && cd doc_vision
docker compose up --build
# API  → http://localhost:8000
# UI   → http://localhost:8501
# Docs → http://localhost:8000/docs
```

Batch processing:
```bash
mkdir input_docs && cp /your/docs/*.pdf input_docs/
docker compose --profile batch up batch
```

### Option 2 — Local Python

```bash
pip install -r requirements.txt
python run.py invoice.pdf              # CLI
uvicorn api.main:app --port 8000       # API
streamlit run ui/dashboard.py          # Dashboard
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET  | /health | Health check |
| POST | /analyze | Full OCR + detection + classification |
| POST | /analyze/summary | Lightweight summary |
| POST | /ocr | Text extraction only |
| POST | /detect | Region detection (images) |

---

## 🏋️ Training

```bash
# Train ML classifier with synthetic data
python training/train_classifier.py --synthetic --output models/classifier.pkl

# Train with real labeled data
python training/train_classifier.py --data-dir ./labeled_docs --algorithm svm --tune

# Fine-tune YOLOv8
python training/train_yolo.py --data dataset.yaml --epochs 50 --imgsz 1024
```

---

## 📦 Batch Processing

```bash
python scripts/batch_process.py \
  --input /path/to/docs/ \
  --output results/ \
  --workers 4 \
  --save-db \
  --export-csv \
  --resume
```

---

## 🧪 Tests

```bash
pytest tests/test_pipeline.py -v
```

---

## 🏗️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Image preprocessing | OpenCV 4.9 |
| OCR | EasyOCR (80+ languages) |
| Object detection | YOLOv8 (Ultralytics) |
| Classification | TF-IDF + SVM / Logistic Regression |
| Entity extraction | Regex patterns |
| PDF rendering | PyMuPDF |
| REST API | FastAPI + Uvicorn |
| Web dashboard | Streamlit + Plotly |
| Database | SQLite with FTS5 full-text search |
| Containers | Docker + docker-compose |
| Testing | pytest (25+ tests) |

---

## 🆕 Added / Fixed (Completion Pass)

| Item | Location | Notes |
|------|----------|-------|
| `.env` | `.env` | Ready-to-use config (copied from `.env.example`) |
| Config file | `config/pipeline_config.yaml` | Central YAML config for all pipeline params |
| YOLO downloader | `scripts/download_yolo_model.py` | Auto-downloads YOLOv8n base weights |
| Sample doc generator | `sample_docs/generate_samples.py` | Generates 5 test images (invoice, receipt, letter, form, resume) |
| `.gitignore` | `.gitignore` | Excludes `.env`, large model `.pt` files, DB, logs |
| Empty folder stubs | `logs/`, `outputs/`, `notebooks/` | `.gitkeep` so Git tracks them |

### ⚡ First-Time Setup (after cloning)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download YOLO base model (optional but recommended)
python scripts/download_yolo_model.py

# 3. Generate sample test documents
python sample_docs/generate_samples.py

# 4. Run the pipeline on a sample
python run.py sample_docs/sample_invoice.png

# 5. Or start the full stack
docker compose up --build
```
