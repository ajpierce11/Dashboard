"""
doc_images.py
Extract embedded images from uploaded study documents so the Report
Generator can send them alongside the extracted text. The text layer is
still produced by doc_text.extract_text — this module is only for the
image side of multimodal prompting.

Returned shape per image:
    {
        "name":       str,   # display label, e.g. "page3_img17.png"
        "media_type": str,   # e.g. "image/png", "image/jpeg"
        "data_b64":   str,   # base64-encoded raw bytes (no data URI prefix)
        "size_bytes": int,
    }

PDF: uses PyMuPDF (fitz) to walk every page and pull each embedded raster
image. Vector graphics and rendered page bitmaps are NOT included — if a
PDF has charts that exist only as drawn paths, those won't appear here.

PPTX / DOCX: both are zip files. Office stores embedded media under
ppt/media/ or word/media/. We just unzip and grab the supported raster
formats. This catches any image the user inserted into a slide or
document, but does NOT render the slide as a picture (so SmartArt,
generated charts, and shapes won't be captured as images).
"""

from __future__ import annotations

import base64
import zipfile
from pathlib import Path

# Per-document caps. Each image consumes ~1.5k input tokens for typical
# slide-sized inputs, so 20 puts the cap around 30k tokens of image data —
# well within Claude's context but expensive enough that we don't want
# unbounded extraction from a 200-page PDF.
MAX_IMAGES_PER_DOC = 20
# Skip absurdly large embedded assets — usually a sign of a print-resolution
# scan that won't help analysis and will blow the request size up.
MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5 MB

_RASTER_EXTS = {"png", "jpg", "jpeg", "gif", "webp"}


def extract_images(file_path: str) -> list[dict]:
    """
    Best-effort image extraction. Returns [] for unsupported file types or
    on any failure — callers should treat it as optional context, never
    rely on a particular image being present.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            return _extract_pdf(path)
        if suffix in (".pptx", ".docx"):
            return _extract_office(path)
    except Exception:
        return []
    return []


def _extract_pdf(path: Path) -> list[dict]:
    import fitz  # PyMuPDF

    out: list[dict] = []
    seen_xrefs: set[int] = set()
    with fitz.open(path) as doc:
        for page_idx, page in enumerate(doc, start=1):
            for img_info in page.get_images(full=True):
                if len(out) >= MAX_IMAGES_PER_DOC:
                    return out
                xref = img_info[0]
                # Same image referenced from multiple pages — skip dupes.
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)
                try:
                    base = doc.extract_image(xref)
                except Exception:
                    continue
                data: bytes = base.get("image") or b""
                if not data or len(data) > MAX_IMAGE_BYTES:
                    continue
                ext = (base.get("ext") or "png").lower()
                if ext == "jpg":
                    media_type = "image/jpeg"
                elif ext in _RASTER_EXTS:
                    media_type = f"image/{ext}"
                else:
                    # Skip exotic formats Claude won't accept (jpx, tiff…)
                    continue
                out.append({
                    "name":       f"page{page_idx}_img{xref}.{ext}",
                    "media_type": media_type,
                    "data_b64":   base64.b64encode(data).decode("ascii"),
                    "size_bytes": len(data),
                })
    return out


def _extract_office(path: Path) -> list[dict]:
    """PPTX and DOCX both store embedded media under <prefix>/media/."""
    out: list[dict] = []
    with zipfile.ZipFile(path) as zf:
        # Sort so ordering is deterministic across runs.
        media_paths = sorted(
            n for n in zf.namelist()
            if "/media/" in n and not n.endswith("/")
        )
        for n in media_paths:
            if len(out) >= MAX_IMAGES_PER_DOC:
                break
            ext = Path(n).suffix.lstrip(".").lower()
            if ext not in _RASTER_EXTS:
                continue
            try:
                data = zf.read(n)
            except Exception:
                continue
            if not data or len(data) > MAX_IMAGE_BYTES:
                continue
            media_type = "image/jpeg" if ext == "jpg" else f"image/{ext}"
            out.append({
                "name":       Path(n).name,
                "media_type": media_type,
                "data_b64":   base64.b64encode(data).decode("ascii"),
                "size_bytes": len(data),
            })
    return out
