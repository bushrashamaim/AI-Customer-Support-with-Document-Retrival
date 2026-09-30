"""
Document CV CLI
Usage:
    python run.py path/to/document.pdf
    python run.py path/to/scan.png --ocr-only
    python run.py path/to/doc.pdf --max-pages 5 --json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from core.pipeline import DocumentPipeline
from loguru import logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Document CV Pipeline — Analyze documents with OCR, detection, and classification.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="Path to PDF or image file")
    parser.add_argument("--max-pages", type=int, default=None, help="Max pages to process (PDF only)")
    parser.add_argument("--json", action="store_true", help="Output full JSON result")
    parser.add_argument("--summary", action="store_true", help="Print summary only")
    parser.add_argument("--ocr-only", action="store_true", help="Print extracted text only")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for OCR/detection")
    parser.add_argument("--languages", nargs="+", default=["en"], help="OCR languages (e.g. en fr de)")
    parser.add_argument("--quiet", action="store_true", help="Suppress log output")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.quiet:
        logger.remove()
        logger.add(sys.stderr, level="WARNING")

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Document CV Pipeline")
    print(f"  File: {input_path.name}")
    print(f"{'='*60}\n")

    pipeline = DocumentPipeline(
        ocr_languages=args.languages,
        use_gpu=args.gpu,
    )

    result = pipeline.process(input_path, max_pages=args.max_pages)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        return

    if args.ocr_only:
        for page in result.pages:
            if len(result.pages) > 1:
                print(f"\n--- Page {page.page_number} ---")
            print(page.ocr.full_text)
        return

    # Default: human-readable summary
    print(f"  Document Type   : {result.document_type.upper()}")
    print(f"  Confidence      : {result.document_confidence:.1%}")
    print(f"  Pages           : {result.page_count}")
    print(f"  Total Words     : {len(result.total_text.split())}")
    print(f"  Processing Time : {result.total_time_ms:.0f} ms")

    if result.metadata:
        print(f"\n  Metadata:")
        for k, v in result.metadata.items():
            if v:
                print(f"    {k:12}: {v}")

    for page in result.pages:
        print(f"\n  {'─'*50}")
        print(f"  Page {page.page_number}")
        print(f"  {'─'*50}")
        print(f"    OCR words       : {len(page.ocr.full_text.split())}")
        print(f"    OCR confidence  : {page.ocr.avg_confidence:.1%}")
        print(f"    Regions found   : {len(page.detections.detections)}")

        if page.detections.detections:
            counts = page.detections.to_dict()["label_counts"]
            print(f"    Region types    : {counts}")

        if page.entities:
            print(f"    Entities found  :")
            for etype, values in page.entities.items():
                print(f"      {etype:20}: {values[:3]}")

        if not args.summary and page.ocr.full_text:
            preview = page.ocr.full_text[:300].replace("\n", " ")
            print(f"\n    Text preview    : {preview}...")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
