#!/usr/bin/env python3
import sys
import os
from PyPDF2 import PdfReader

DEFAULT_PDF_PATH = r"C:\Users\admin\PycharmProjects\model2026\tentative report.pdf"


def extract_text_from_pdf(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"PDF not found: {path}")
    texts = []
    with open(path, "rb") as f:
        reader = PdfReader(f)
        num_pages = len(reader.pages)
        for i, page in enumerate(reader.pages, start=1):
            try:
                t = page.extract_text()
            except Exception:
                t = None
            texts.append(t or "")
    return "\n\n".join(texts)


def main():
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PDF_PATH
    out_path = os.path.join(os.path.dirname(__file__), "..", "tentative_report.txt")
    out_path = os.path.abspath(out_path)

    try:
        text = extract_text_from_pdf(pdf_path)
    except Exception as e:
        print("ERROR: Failed to extract text:", e)
        sys.exit(2)

    # Write output
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

    if not text.strip():
        print("No extractable text was found in the PDF. It may be a scanned/image PDF that requires OCR.")
        print(f"Wrote output (empty or minimal) to: {out_path}")
        sys.exit(0)

    print(f"Extracted text written to: {out_path}\n")
    print("--- Begin extracted text ---")
    print(text)
    print("--- End extracted text ---")


if __name__ == "__main__":
    main()
