"""
Generate sample test documents (PNG images) for pipeline testing.
Creates one sample per supported document type.

Usage:
    python sample_docs/generate_samples.py
"""

from pathlib import Path
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

OUTPUT_DIR = Path(__file__).parent


def make_canvas(w=794, h=1123):
    """A4-ish white canvas."""
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    return img, draw


def try_font(size=14):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def generate_invoice():
    img, draw = make_canvas()
    f_big, f_med, f_sm = try_font(22), try_font(14), try_font(11)
    # Header
    draw.rectangle([40, 40, 754, 90], fill="#1a4f8a")
    draw.text((50, 52), "INVOICE", fill="white", font=f_big)
    draw.text((580, 52), "INV-2024-00123", fill="white", font=f_med)
    # Bill to
    draw.text((50, 110), "Bill To:", fill="#333", font=f_med)
    draw.text((50, 130), "Acme Corp, 123 Business Ave, Karachi", fill="#555", font=f_sm)
    draw.text((50, 148), "Date: 2024-05-14    Due: 2024-06-14", fill="#555", font=f_sm)
    # Table header
    draw.rectangle([40, 190, 754, 215], fill="#e8f0fe")
    for col, label in [(50, "Description"), (400, "Qty"), (500, "Rate"), (640, "Amount")]:
        draw.text((col, 196), label, fill="#1a4f8a", font=f_med)
    # Rows
    rows = [
        ("OCR Processing Service (100 pages)", "1", "$200.00", "$200.00"),
        ("Document Classification Module", "2", "$150.00", "$300.00"),
        ("API Integration Setup", "1", "$350.00", "$350.00"),
    ]
    y = 220
    for desc, qty, rate, amt in rows:
        draw.text((50, y), desc, fill="#333", font=f_sm)
        draw.text((400, y), qty, fill="#333", font=f_sm)
        draw.text((500, y), rate, fill="#333", font=f_sm)
        draw.text((640, y), amt, fill="#333", font=f_sm)
        draw.line([(40, y + 20), (754, y + 20)], fill="#ddd")
        y += 30
    # Total
    draw.rectangle([500, y + 10, 754, y + 40], fill="#1a4f8a")
    draw.text((510, y + 17), "TOTAL:   $850.00", fill="white", font=f_med)
    img.save(OUTPUT_DIR / "sample_invoice.png")
    print("✅ sample_invoice.png")


def generate_receipt():
    img, draw = make_canvas(400, 600)
    f_big, f_med, f_sm = try_font(18), try_font(13), try_font(10)
    draw.text((150, 30), "RECEIPT", fill="#111", font=f_big)
    draw.text((120, 58), "SuperMart Store #42", fill="#555", font=f_sm)
    draw.text((130, 73), "Karachi, Pakistan", fill="#555", font=f_sm)
    draw.line([(20, 95), (380, 95)], fill="#999")
    items = [("Milk 1L", "120"), ("Bread", "80"), ("Eggs x12", "240"),
             ("Butter 500g", "350"), ("Orange Juice", "195")]
    y = 110
    for name, price in items:
        draw.text((25, y), name, fill="#333", font=f_sm)
        draw.text((330, y), f"Rs {price}", fill="#333", font=f_sm)
        y += 22
    draw.line([(20, y + 5), (380, y + 5)], fill="#999")
    draw.text((25, y + 15), "TOTAL:", fill="#111", font=f_med)
    draw.text((300, y + 15), "Rs 985", fill="#111", font=f_med)
    draw.text((100, y + 50), "Thank you for shopping!", fill="#777", font=f_sm)
    img.save(OUTPUT_DIR / "sample_receipt.png")
    print("✅ sample_receipt.png")


def generate_letter():
    img, draw = make_canvas()
    f_med, f_sm = try_font(13), try_font(11)
    draw.text((50, 50), "TechCorp Pakistan (Pvt) Ltd.", fill="#111", font=f_med)
    draw.text((50, 70), "Tower A, Business Hub, Karachi", fill="#555", font=f_sm)
    draw.text((50, 130), "14 May, 2024", fill="#333", font=f_sm)
    draw.text((50, 170), "To Whom It May Concern,", fill="#111", font=f_sm)
    body = (
        "We are pleased to inform you that your application for the Document\n"
        "Processing Integration project has been approved. Our team has reviewed\n"
        "all submitted materials and found them to be in order.\n\n"
        "Please proceed to the next phase of onboarding by submitting the\n"
        "required documentation to our procurement department at your earliest\n"
        "convenience. The project kick-off is scheduled for June 1st, 2024.\n\n"
        "Should you have any questions, do not hesitate to contact us."
    )
    y = 210
    for line in body.split("\n"):
        draw.text((50, y), line, fill="#333", font=f_sm)
        y += 20
    draw.text((50, y + 30), "Sincerely,", fill="#333", font=f_sm)
    draw.text((50, y + 70), "Ali Hassan — Director of Operations", fill="#111", font=f_med)
    img.save(OUTPUT_DIR / "sample_letter.png")
    print("✅ sample_letter.png")


def generate_form():
    img, draw = make_canvas()
    f_big, f_med, f_sm = try_font(20), try_font(13), try_font(11)
    draw.rectangle([40, 30, 754, 80], fill="#2e7d32")
    draw.text((50, 44), "REGISTRATION FORM", fill="white", font=f_big)
    fields = [
        ("Full Name", 120), ("Date of Birth", 180), ("CNIC Number", 240),
        ("Email Address", 300), ("Phone Number", 360), ("Home Address", 420),
    ]
    for label, y in fields:
        draw.text((50, y), label + ":", fill="#333", font=f_med)
        draw.rectangle([200, y - 2, 700, y + 22], outline="#999")
    # Checkboxes
    draw.text((50, 490), "Gender:", fill="#333", font=f_med)
    for i, opt in enumerate(["Male", "Female", "Other"]):
        x = 200 + i * 120
        draw.rectangle([x, 488, x + 16, 504], outline="#333")
        draw.text((x + 22, 490), opt, fill="#333", font=f_sm)
    draw.rectangle([40, 560, 754, 600], fill="#2e7d32")
    draw.text((300, 572), "SUBMIT FORM", fill="white", font=f_med)
    img.save(OUTPUT_DIR / "sample_form.png")
    print("✅ sample_form.png")


def generate_resume():
    img, draw = make_canvas()
    f_big, f_med, f_sm = try_font(20), try_font(13), try_font(11)
    draw.rectangle([0, 0, 794, 130], fill="#1a237e")
    draw.text((50, 30), "SARA KHAN", fill="white", font=f_big)
    draw.text((50, 65), "Software Engineer  |  sara.khan@email.com  |  +92-300-1234567", fill="#c5cae9", font=f_sm)
    draw.text((50, 85), "linkedin.com/in/sarakhan  |  github.com/sarakhan", fill="#c5cae9", font=f_sm)
    sections = {
        "EXPERIENCE": [
            "Senior Dev — DocuTech  (2022–Present)",
            "  • Built OCR pipeline processing 50K docs/day",
            "  • Led team of 4 engineers",
            "",
            "Software Dev — DataSoft  (2019–2022)",
            "  • REST API development with FastAPI",
            "  • Maintained ML model deployment",
        ],
        "EDUCATION": [
            "B.Sc. Computer Science — FAST-NUCES Karachi  (2019)",
            "GPA: 3.8 / 4.0",
        ],
        "SKILLS": [
            "Python, FastAPI, OpenCV, YOLOv8, EasyOCR, Docker",
            "PostgreSQL, SQLite, Redis, AWS S3",
        ],
    }
    y = 150
    for section, lines in sections.items():
        draw.rectangle([40, y, 754, y + 24], fill="#e8eaf6")
        draw.text((50, y + 5), section, fill="#1a237e", font=f_med)
        y += 32
        for line in lines:
            draw.text((55, y), line, fill="#333", font=f_sm)
            y += 18
        y += 10
    img.save(OUTPUT_DIR / "sample_resume.png")
    print("✅ sample_resume.png")


if __name__ == "__main__":
    if not PIL_AVAILABLE:
        print("❌ Pillow not installed. Run: pip install Pillow")
        exit(1)
    print("Generating sample documents...")
    generate_invoice()
    generate_receipt()
    generate_letter()
    generate_form()
    generate_resume()
    print(f"\n✅ All samples saved to: {OUTPUT_DIR}")
