"""
Batch Document Processor
Process entire folders of PDFs and images with progress tracking,
parallel execution, error recovery, and CSV/JSON export.

Usage:
    python scripts/batch_process.py --input /path/to/docs --output results/
    python scripts/batch_process.py --input /docs --workers 4 --export-csv
    python scripts/batch_process.py --input /docs --save-db --resume
"""

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.pipeline import DocumentPipeline


SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}


# ─── Progress tracker ─────────────────────────────────────────────────────────

class BatchProgress:
    def __init__(self, total: int):
        self.total = total
        self.done = 0
        self.failed = 0
        self.skipped = 0
        self.start_time = time.perf_counter()

    def update(self, success: bool, skipped: bool = False):
        if skipped:
            self.skipped += 1
        elif success:
            self.done += 1
        else:
            self.failed += 1

    def eta_str(self) -> str:
        elapsed = time.perf_counter() - self.start_time
        processed = self.done + self.failed
        if processed == 0:
            return "calculating..."
        rate = processed / elapsed
        remaining = (self.total - processed) / rate
        mins, secs = divmod(int(remaining), 60)
        return f"{mins}m {secs}s"

    def bar(self) -> str:
        processed = self.done + self.failed + self.skipped
        pct = processed / self.total * 100 if self.total else 0
        filled = int(pct / 5)
        bar = "█" * filled + "░" * (20 - filled)
        return (
            f"[{bar}] {pct:.1f}% | "
            f"✓{self.done} ✗{self.failed} ↷{self.skipped} | "
            f"ETA: {self.eta_str()}"
        )


# ─── Worker function ──────────────────────────────────────────────────────────

def process_single(
    file_path: Path,
    pipeline: DocumentPipeline,
    max_pages: Optional[int] = None,
    output_dir: Optional[Path] = None,
    save_db: bool = False,
    db=None,
) -> dict:
    """Process a single file and return result dict."""
    t0 = time.perf_counter()
    try:
        result = pipeline.process(file_path, max_pages=max_pages)

        output = {
            "file": str(file_path),
            "file_name": file_path.name,
            "status": "ok",
            "document_type": result.document_type,
            "confidence": round(result.document_confidence, 4),
            "page_count": result.page_count,
            "word_count": len(result.total_text.split()),
            "processing_ms": round(result.total_time_ms, 1),
        }

        # Collect all entities across pages
        all_entities = {}
        for page in result.pages:
            for etype, vals in page.entities.items():
                all_entities.setdefault(etype, []).extend(vals)
        output["entities"] = {k: list(set(v)) for k, v in all_entities.items()}

        # Save JSON result
        if output_dir:
            out_file = output_dir / f"{file_path.stem}_result.json"
            out_file.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))

        # Save to DB
        if save_db and db:
            doc_id = db.save_result(result)
            output["db_id"] = doc_id

        return output

    except Exception as e:
        logger.error(f"Failed: {file_path.name} — {e}")
        return {
            "file": str(file_path),
            "file_name": file_path.name,
            "status": "error",
            "error": str(e),
            "processing_ms": round((time.perf_counter() - t0) * 1000, 1),
        }


# ─── Batch runner ─────────────────────────────────────────────────────────────

class BatchProcessor:
    def __init__(
        self,
        pipeline: DocumentPipeline,
        output_dir: Optional[str] = None,
        max_pages: Optional[int] = None,
        workers: int = 2,
        save_db: bool = False,
        resume: bool = False,
    ):
        self.pipeline = pipeline
        self.output_dir = Path(output_dir) if output_dir else None
        self.max_pages = max_pages
        self.workers = workers
        self.save_db = save_db
        self.resume = resume
        self.db = None

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        if save_db:
            from database.db_manager import DocumentDB
            self.db = DocumentDB()

    def collect_files(self, input_path: Path) -> list[Path]:
        """Recursively collect all supported documents."""
        if input_path.is_file():
            return [input_path] if input_path.suffix.lower() in SUPPORTED_EXTENSIONS else []

        files = []
        for ext in SUPPORTED_EXTENSIONS:
            files.extend(input_path.rglob(f"*{ext}"))
            files.extend(input_path.rglob(f"*{ext.upper()}"))
        return sorted(set(files))

    def _is_done(self, file_path: Path) -> bool:
        """Check if file was already processed (for resume mode)."""
        if not self.resume or not self.output_dir:
            return False
        return (self.output_dir / f"{file_path.stem}_result.json").exists()

    def run(self, input_path: str) -> list[dict]:
        """Run batch processing on a directory or file."""
        files = self.collect_files(Path(input_path))
        if not files:
            logger.warning(f"No supported files found in: {input_path}")
            return []

        logger.info(f"Found {len(files)} documents to process (workers={self.workers})")

        progress = BatchProgress(len(files))
        results = []

        if self.workers == 1:
            # Sequential (safer for GPU/memory-constrained environments)
            for i, fpath in enumerate(files):
                if self._is_done(fpath):
                    progress.update(True, skipped=True)
                    logger.debug(f"Skipping (already done): {fpath.name}")
                    continue

                result = process_single(
                    fpath, self.pipeline, self.max_pages,
                    self.output_dir, self.save_db, self.db
                )
                results.append(result)
                progress.update(result["status"] == "ok")
                print(f"\r{progress.bar()}  {fpath.name[:40]:<40}", end="", flush=True)
        else:
            # Parallel with ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                future_to_file = {}
                for fpath in files:
                    if self._is_done(fpath):
                        progress.update(True, skipped=True)
                        continue
                    future = executor.submit(
                        process_single, fpath, self.pipeline,
                        self.max_pages, self.output_dir, self.save_db, self.db
                    )
                    future_to_file[future] = fpath

                for future in as_completed(future_to_file):
                    fpath = future_to_file[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        result = {"file": str(fpath), "status": "error", "error": str(e)}
                    results.append(result)
                    progress.update(result["status"] == "ok")
                    print(f"\r{progress.bar()}  {fpath.name[:40]:<40}", end="", flush=True)

        print()  # newline after progress bar
        self._print_summary(results, progress)
        return results

    def _print_summary(self, results: list[dict], progress: BatchProgress):
        elapsed = time.perf_counter() - progress.start_time
        ok = [r for r in results if r.get("status") == "ok"]
        errors = [r for r in results if r.get("status") == "error"]

        print(f"\n{'='*60}")
        print(f"  BATCH PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"  Total files    : {progress.total}")
        print(f"  Processed      : {len(ok)}")
        print(f"  Errors         : {len(errors)}")
        print(f"  Skipped        : {progress.skipped}")
        print(f"  Total time     : {elapsed:.1f}s ({elapsed/max(len(ok),1):.1f}s/doc avg)")
        print()

        if ok:
            from collections import Counter
            type_dist = Counter(r.get("document_type", "unknown") for r in ok)
            print("  Document Types:")
            for dtype, count in type_dist.most_common():
                bar = "▓" * count
                print(f"    {dtype:20}: {count:4}  {bar}")
            print()

        if errors:
            print("  Errors:")
            for e in errors[:5]:
                print(f"    ✗ {e['file_name']}: {e.get('error', '')}")
            if len(errors) > 5:
                print(f"    ... and {len(errors)-5} more")

    def export_csv(self, results: list[dict], output_path: str):
        """Export flat CSV summary of all results."""
        if not results:
            return
        flat_keys = ["file_name", "status", "document_type", "confidence",
                     "page_count", "word_count", "processing_ms", "error"]
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=flat_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        logger.success(f"CSV exported: {output_path}")

    def export_json(self, results: list[dict], output_path: str):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.success(f"JSON exported: {output_path}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Batch process document folder")
    p.add_argument("--input", required=True, help="Input directory or file")
    p.add_argument("--output", default="batch_results/", help="Output directory for JSON results")
    p.add_argument("--workers", type=int, default=1, help="Parallel workers (1=sequential)")
    p.add_argument("--max-pages", type=int, default=None, help="Max pages per PDF")
    p.add_argument("--save-db", action="store_true", help="Save results to SQLite DB")
    p.add_argument("--resume", action="store_true", help="Skip already-processed files")
    p.add_argument("--export-csv", action="store_true", help="Export summary CSV")
    p.add_argument("--export-json", action="store_true", help="Export summary JSON")
    p.add_argument("--gpu", action="store_true", help="Use GPU")
    p.add_argument("--languages", nargs="+", default=["en"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    pipeline = DocumentPipeline(ocr_languages=args.languages, use_gpu=args.gpu)
    processor = BatchProcessor(
        pipeline=pipeline,
        output_dir=args.output,
        max_pages=args.max_pages,
        workers=args.workers,
        save_db=args.save_db,
        resume=args.resume,
    )

    results = processor.run(args.input)

    if args.export_csv:
        processor.export_csv(results, Path(args.output) / "batch_summary.csv")
    if args.export_json:
        processor.export_json(results, Path(args.output) / "batch_summary.json")
