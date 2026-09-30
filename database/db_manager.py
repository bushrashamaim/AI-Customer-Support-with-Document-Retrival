"""
SQLite Database Manager
Stores document analysis results, enables search, history, and reporting.

Schema:
  documents  — one row per uploaded document
  pages      — one row per page
  entities   — extracted entities per page
  detections — detected regions per page
"""

import json
import sqlite3
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional
from loguru import logger
from contextlib import contextmanager


DB_PATH = Path(__file__).parent.parent / "database" / "doc_cv.db"


# ─── Schema ───────────────────────────────────────────────────────────────────

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS documents (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    file_hash       TEXT UNIQUE NOT NULL,
    file_name       TEXT NOT NULL,
    file_type       TEXT NOT NULL,
    document_type   TEXT,
    confidence      REAL,
    page_count      INTEGER DEFAULT 1,
    word_count      INTEGER DEFAULT 0,
    full_text       TEXT,
    metadata        TEXT,              -- JSON
    processing_ms   REAL,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id     INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_number     INTEGER NOT NULL,
    ocr_text        TEXT,
    word_count      INTEGER DEFAULT 0,
    ocr_confidence  REAL,
    doc_type        TEXT,
    doc_confidence  REAL,
    processing_ms   REAL,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS entities (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    page_id         INTEGER NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
    entity_type     TEXT NOT NULL,
    entity_value    TEXT NOT NULL,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS detections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    page_id         INTEGER NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
    label           TEXT NOT NULL,
    confidence      REAL,
    x INTEGER, y INTEGER, w INTEGER, h INTEGER,
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_docs_type     ON documents(document_type);
CREATE INDEX IF NOT EXISTS idx_docs_hash     ON documents(file_hash);
CREATE INDEX IF NOT EXISTS idx_docs_created  ON documents(created_at);
CREATE INDEX IF NOT EXISTS idx_pages_doc     ON pages(document_id);
CREATE INDEX IF NOT EXISTS idx_entities_page ON entities(page_id);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_detections_pg ON detections(page_id);
CREATE INDEX IF NOT EXISTS idx_detections_lb ON detections(label);

CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts
USING fts5(file_name, full_text, document_type, content=documents, content_rowid=id);

CREATE TRIGGER IF NOT EXISTS docs_ai AFTER INSERT ON documents BEGIN
    INSERT INTO docs_fts(rowid, file_name, full_text, document_type)
    VALUES (new.id, new.file_name, new.full_text, new.document_type);
END;

CREATE TRIGGER IF NOT EXISTS docs_au AFTER UPDATE ON documents BEGIN
    UPDATE docs_fts SET
        file_name=new.file_name, full_text=new.full_text, document_type=new.document_type
    WHERE rowid=new.id;
END;

CREATE TRIGGER IF NOT EXISTS docs_ad AFTER DELETE ON documents BEGIN
    DELETE FROM docs_fts WHERE rowid=old.id;
END;
"""


# ─── Manager ─────────────────────────────────────────────────────────────────

class DocumentDB:
    """
    SQLite-backed document result store.
    Thread-safe via connection-per-call pattern.
    """

    def __init__(self, db_path: str = None):
        self.db_path = Path(db_path or DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info(f"DocumentDB initialized: {self.db_path}")

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript(SCHEMA_SQL)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ─── Write ───────────────────────────────────────────────────────────────

    def save_result(self, result) -> int:
        """
        Persist a DocumentResult (from pipeline.py) to the database.
        Returns the document ID. Skips if duplicate (by file hash).
        """
        file_hash = self._hash_text(result.total_text + result.file_name)
        now = datetime.utcnow().isoformat()

        with self._conn() as conn:
            # Upsert document
            existing = conn.execute(
                "SELECT id FROM documents WHERE file_hash=?", (file_hash,)
            ).fetchone()

            if existing:
                doc_id = existing["id"]
                conn.execute(
                    """UPDATE documents SET
                        document_type=?, confidence=?, word_count=?,
                        processing_ms=?, updated_at=?
                       WHERE id=?""",
                    (result.document_type, result.document_confidence,
                     len(result.total_text.split()), result.total_time_ms, now, doc_id)
                )
                logger.debug(f"Updated existing document id={doc_id}")
            else:
                cur = conn.execute(
                    """INSERT INTO documents
                        (file_hash, file_name, file_type, document_type, confidence,
                         page_count, word_count, full_text, metadata, processing_ms, created_at, updated_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (file_hash, result.file_name, result.file_type,
                     result.document_type, result.document_confidence,
                     result.page_count, len(result.total_text.split()),
                     result.total_text[:50000],  # cap to 50k chars
                     json.dumps(result.metadata),
                     result.total_time_ms, now, now)
                )
                doc_id = cur.lastrowid
                logger.info(f"Saved new document id={doc_id}: {result.file_name}")

            # Save pages
            conn.execute("DELETE FROM pages WHERE document_id=?", (doc_id,))
            for page in result.pages:
                page_cur = conn.execute(
                    """INSERT INTO pages
                        (document_id, page_number, ocr_text, word_count,
                         ocr_confidence, doc_type, doc_confidence, processing_ms, created_at)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    (doc_id, page.page_number, page.ocr.full_text,
                     len(page.ocr.full_text.split()), page.ocr.avg_confidence,
                     page.classification.predicted_type, page.classification.confidence,
                     page.processing_time_ms, now)
                )
                page_id = page_cur.lastrowid

                # Save entities
                for etype, values in page.entities.items():
                    for val in values:
                        conn.execute(
                            "INSERT INTO entities (page_id, entity_type, entity_value, created_at) VALUES (?,?,?,?)",
                            (page_id, etype, str(val), now)
                        )

                # Save detections
                for det in page.detections.detections:
                    conn.execute(
                        """INSERT INTO detections
                            (page_id, label, confidence, x, y, w, h, created_at)
                           VALUES (?,?,?,?,?,?,?,?)""",
                        (page_id, det.label, det.confidence,
                         det.x, det.y, det.width, det.height, now)
                    )

        return doc_id

    # ─── Read ────────────────────────────────────────────────────────────────

    def get_document(self, doc_id: int) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
            if not row:
                return None
            doc = dict(row)
            doc["metadata"] = json.loads(doc["metadata"] or "{}")
            doc["pages"] = self._get_pages(conn, doc_id)
            return doc

    def _get_pages(self, conn, doc_id: int) -> list[dict]:
        pages = conn.execute(
            "SELECT * FROM pages WHERE document_id=? ORDER BY page_number", (doc_id,)
        ).fetchall()
        result = []
        for page in pages:
            p = dict(page)
            p["entities"] = self._get_entities(conn, p["id"])
            p["detections"] = self._get_detections(conn, p["id"])
            result.append(p)
        return result

    def _get_entities(self, conn, page_id: int) -> dict:
        rows = conn.execute(
            "SELECT entity_type, entity_value FROM entities WHERE page_id=?", (page_id,)
        ).fetchall()
        entities = {}
        for row in rows:
            entities.setdefault(row["entity_type"], []).append(row["entity_value"])
        return entities

    def _get_detections(self, conn, page_id: int) -> list[dict]:
        rows = conn.execute(
            "SELECT label, confidence, x, y, w, h FROM detections WHERE page_id=?", (page_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def list_documents(
        self,
        limit: int = 50,
        offset: int = 0,
        doc_type: str = None,
        since: str = None,
    ) -> list[dict]:
        """List documents with optional filters."""
        query = "SELECT id, file_name, file_type, document_type, confidence, page_count, word_count, processing_ms, created_at FROM documents"
        params = []
        conditions = []

        if doc_type:
            conditions.append("document_type=?")
            params.append(doc_type)
        if since:
            conditions.append("created_at >= ?")
            params.append(since)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params += [limit, offset]

        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def search(self, query: str, limit: int = 20) -> list[dict]:
        """Full-text search across document text and file names."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT d.id, d.file_name, d.document_type, d.confidence, d.created_at,
                          snippet(docs_fts, 1, '<b>', '</b>', '...', 20) AS snippet
                   FROM docs_fts
                   JOIN documents d ON docs_fts.rowid = d.id
                   WHERE docs_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (query, limit)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_stats(self) -> dict:
        """Aggregated stats for dashboard."""
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            by_type = conn.execute(
                "SELECT document_type, COUNT(*) as count FROM documents GROUP BY document_type ORDER BY count DESC"
            ).fetchall()
            recent = conn.execute(
                "SELECT file_name, document_type, confidence, created_at FROM documents ORDER BY created_at DESC LIMIT 5"
            ).fetchall()
            avg_conf = conn.execute("SELECT AVG(confidence) FROM documents WHERE confidence > 0").fetchone()[0]
            avg_time = conn.execute("SELECT AVG(processing_ms) FROM documents").fetchone()[0]
            entity_counts = conn.execute(
                "SELECT entity_type, COUNT(*) as count FROM entities GROUP BY entity_type ORDER BY count DESC"
            ).fetchall()
            detection_counts = conn.execute(
                "SELECT label, COUNT(*) as count FROM detections GROUP BY label ORDER BY count DESC"
            ).fetchall()

        return {
            "total_documents": total,
            "avg_confidence": round(avg_conf or 0, 4),
            "avg_processing_ms": round(avg_time or 0, 1),
            "by_type": [dict(r) for r in by_type],
            "recent": [dict(r) for r in recent],
            "entity_type_counts": [dict(r) for r in entity_counts],
            "detection_label_counts": [dict(r) for r in detection_counts],
        }

    def delete_document(self, doc_id: int) -> bool:
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))
            return cur.rowcount > 0

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:32]
