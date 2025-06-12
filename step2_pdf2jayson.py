import os
import json
import fitz  # PyMuPDF
import datetime
import pytesseract
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
from collections import Counter

PDF_ROOT = "policy_data"
JSON_ROOT = "policy_json"

def extract_metadata(pdf_path):
    stat = os.stat(pdf_path)
    try:
        page_count = len(PdfReader(pdf_path).pages)
    except Exception:
        page_count = None
    return {
        "filename": os.path.basename(pdf_path),
        "filepath": pdf_path,
        "size_bytes": stat.st_size,
        "created_time": datetime.datetime.fromtimestamp(stat.st_ctime).isoformat(),
        "modified_time": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "page_count": page_count,
    }

def remove_boilerplate(text):
    pages = text.split("\f")
    line_freq = Counter()
    for page in pages:
        for line in page.strip().split("\n"):
            line_freq[line.strip()] += 1

    boiler_lines = {line for line, freq in line_freq.items() if freq > 2 and len(line) > 5}
    cleaned = []
    for page in pages:
        lines = [
            line for line in page.strip().split("\n")
            if line.strip() not in boiler_lines
        ]
        cleaned.append("\n".join(lines))
    return "\n\n".join(cleaned)

def extract_text_with_fitz(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = "\f".join(page.get_text() for page in doc)
        doc.close()
        return remove_boilerplate(text)
    except Exception as e:
        return ""

def extract_text_with_ocr(pdf_path):
    try:
        images = convert_from_path(pdf_path, dpi=300)
        text_pages = []
        for image in images:
            text = pytesseract.image_to_string(image)
            text_pages.append(text)
        full_text = "\f".join(text_pages)
        return remove_boilerplate(full_text)
    except Exception as e:
        return f"OCR Failed: {str(e)}"

def convert_pdf_to_json(pdf_path, json_path):
    metadata = extract_metadata(pdf_path)
    text = extract_text_with_fitz(pdf_path)
    
    if not text or len(text.strip()) < 100:
        text = extract_text_with_ocr(pdf_path)

    data = {
        "metadata": metadata,
        "content": text
    }

    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def walk_and_convert(pdf_root, json_root):
    for root, _, files in os.walk(pdf_root):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                rel_path = os.path.relpath(pdf_path, pdf_root)
                json_path = os.path.join(json_root, rel_path).replace('.pdf', '.json')
                try:
                    convert_pdf_to_json(pdf_path, json_path)
                    print(f"✓ Converted: {pdf_path}")
                except Exception as e:
                    print(f"✗ Failed: {pdf_path} — {e}")

if __name__ == "__main__":
    walk_and_convert(PDF_ROOT, JSON_ROOT)