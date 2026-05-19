"""
doc_text.py
Shared helper for extracting plain text from library documents.
Supports .pdf (pymupdf → pypdf fallback), .docx, .pptx/.ppt.
"""

from pathlib import Path

SUPPORTED_EXTS = {".pdf", ".docx", ".pptx", ".ppt"}


def extract_text(filepath: str) -> str:
    """Extract complete text from a file regardless of size.

    Returns an empty string if the file does not exist, the extension
    is unsupported, or all extraction paths fail.
    """
    p = Path(filepath)
    if not p.exists():
        return ""
    ext = p.suffix.lower()

    if ext == ".docx":
        try:
            # Some study docs have unusually large inline XML attributes
            # (embedded equations, signature images, OLE blobs) that trip
            # lxml's default 10 MB AttValue limit. Switch the global lxml
            # parser to huge_tree mode before opening the doc — required
            # to read AGN report files that exceed the default limit.
            try:
                from lxml import etree as _etree
                _etree.set_default_parser(
                    _etree.XMLParser(huge_tree=True)
                )
            except Exception:
                pass

            from docx import Document
            doc = Document(str(p))
            parts = []
            for para in doc.paragraphs:
                text = para.text.strip()
                if not text:
                    continue
                if para.style.name.startswith("Heading"):
                    parts.append(f"\n## {text}\n")
                else:
                    parts.append(text)
            return "\n".join(parts)
        except Exception as e:
            print(f"    ! docx error {p.name}: {e}")
            return ""

    if ext == ".pdf":
        # pymupdf first — handles DocuSign/Adobe Sign encrypted PDFs
        try:
            import fitz  # pymupdf
            doc = fitz.open(str(p))
            pages = []
            for page in doc:
                text = page.get_text() or ""
                if text.strip():
                    pages.append(text)
            doc.close()
            if pages:
                return "\n\n".join(pages)
        except ImportError:
            pass
        except Exception:
            pass
        try:
            import pypdf
            reader = pypdf.PdfReader(str(p))
            pages = []
            for page in reader.pages:
                text = page.extract_text() or ""
                if text.strip():
                    pages.append(text)
            if pages:
                return "\n\n".join(pages)
        except Exception as e:
            print(f"    ! pdf error {p.name}: {e}")
        return ""

    if ext in (".pptx", ".ppt"):
        try:
            from pptx import Presentation
            prs = Presentation(str(p))
            slides = []
            for i, slide in enumerate(prs.slides):
                parts = [shape.text.strip() for shape in slide.shapes
                         if hasattr(shape, "text") and shape.text.strip()]
                if parts:
                    slides.append(f"[Slide {i+1}] " + " ".join(parts))
            return "\n".join(slides)
        except Exception as e:
            print(f"    ! pptx error {p.name}: {e}")
            return ""

    return ""
