# document_loader.py
import os
import time
from typing import List, Dict, Any

try:
    from pypdf import PdfReader  # newer name
except Exception:
    PdfReader = None

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def save_uploaded_pdfs(uploaded_files) -> List[str]:
    """
    Save Streamlit uploaded file objects into uploads/ and return list of file paths.
    """
    saved = []
    for f in uploaded_files:
        # streamlit's UploadedFile has .name and .getbuffer()
        name = f.name
        timestamp = int(time.time() * 1000)
        safe_name = f"{timestamp}_{name}"
        path = os.path.join(UPLOAD_DIR, safe_name)
        with open(path, "wb") as fh:
            fh.write(f.getbuffer())
        saved.append(path)
    return saved


def load_pdfs_from_paths(paths: List[str]) -> List[Dict[str, Any]]:
    """
    Read PDFs page-by-page and return list of dicts with keys: 'text','source','page'
    """
    if PdfReader is None:
        raise RuntimeError("Pdf reading library not installed. Install 'pypdf' or 'PyPDF2'.")

    pages = []
    for p in paths:
        try:
            reader = PdfReader(p)
            num_pages = len(reader.pages)
            for i in range(num_pages):
                try:
                    page = reader.pages[i]
                    text = page.extract_text() or ""
                except Exception:
                    text = ""
                pages.append({"text": text, "source": os.path.basename(p), "page": i + 1})
        except Exception as e:
            # skip problematic PDF but surface the error upstream
            raise RuntimeError(f"Failed to read PDF {p}: {e}")
    return pages


def detect_scanned(pages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Lightweight scanned check. If many pages have <30 chars, warn.
    """
    if not pages:
        return {"likely_scanned": True, "empty_ratio": 1.0, "message": "No pages found.", "total_pages": 0, "empty_pages": 0}

    total = len(pages)
    empty = sum(1 for pg in pages if len((pg.get("text") or "").strip()) < 30)
    ratio = empty / max(total, 1)
    likely = ratio >= 0.60
    msg = ("⚠️ This PDF looks like scanned/image-based PDF (low extractable text)."
           if likely else "✅ PDF text looks good (text-based).")
    return {
        "likely_scanned": likely,
        "empty_ratio": ratio,
        "message": msg,
        "total_pages": total,
        "empty_pages": empty
    }
