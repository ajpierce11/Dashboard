"""
ai_metadata.py
AI-extracted scientific metadata layer over the existing library index.

Sits alongside library_index.json and library_vectors.npz — does not
replace them. The existing index tracks files; this tracks *facts* about
the studies (product(s) tested, model type, endpoints, timepoints) so
forward-looking features (coverage matrix, gap analysis, methodology
navigator) can operate on structured fields instead of raw text.

Extraction runs once per study on the content at that time. Re-runs are
incremental — only entries whose file mtimes changed since the last
extraction are re-processed, keeping subsequent refreshes fast.

The prompt is deliberately minimal ("product, model, endpoints,
timepoints"). Richer fields can be added later without invalidating
prior extractions — this schema just grows.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import iliad_client

AI_METADATA_FILE = "library_ai_metadata.json"
SCHEMA_VERSION = 1

# How many parallel extraction workers. Each is a blocking HTTPS request
# through iliad_client, so threads work fine. 10 gives roughly 10x speedup
# over sequential without obviously abusing the gateway.
EXTRACTION_WORKERS = 10

# Per-study text cap. Enough to cover abstract + methods + results for a
# typical study; trimming keeps the per-call token count predictable.
EXTRACTION_TEXT_CAP = 12000

EXTRACTION_SYSTEM_PROMPT = """You are extracting structured metadata from a pre-clinical dermal filler study written by an AbbVie research team.

From the provided study text, extract exactly these four fields:

- product: the main product(s) or experimental formulation(s) tested. Use the internal name if present (e.g. "Voluma", "Harmonyca", "AGN-XYZ"). If multiple, comma-separate. Empty string if no specific product is tested (e.g. a general test method document).
- model: the experimental model or system used, short form. Examples: "in vitro", "ex vivo human skin", "rat dorsal subcutaneous", "mouse subcutaneous", "clinical". Use the most specific descriptor present. Empty string if the document is not an experimental study.
- endpoints: a list of the primary measured endpoints. Examples: "lift capacity", "elasticity (G')", "cohesivity", "water uptake", "collagen I expression", "CD68 macrophage infiltration", "histology score". Empty list if none apply.
- timepoints: a list of the timepoints measured, in short form like "0w", "4w", "12w", "24w", "52w" (weeks) or "1d", "30d" (days). Empty list if not time-resolved or not stated.

Output ONLY a JSON object with these four fields. No markdown fences, no preamble, no trailing text. If a field is genuinely unclear from the text, use "" or [] rather than guessing."""


def metadata_file_path(library_path: str | Path) -> Path:
    return Path(library_path) / AI_METADATA_FILE


def load_metadata(library_path: str | Path) -> dict:
    """
    Load the AI metadata cache. Returns the parsed JSON or a fresh empty
    shell. Never raises on bad JSON — corrupt files are treated as empty
    so a downstream extraction pass recreates them cleanly.
    """
    path = metadata_file_path(library_path)
    if not path.exists():
        return {"version": SCHEMA_VERSION, "last_updated": "", "entries": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "entries" not in data:
            raise ValueError("unexpected shape")
        data.setdefault("version", SCHEMA_VERSION)
        data.setdefault("last_updated", "")
        return data
    except Exception:
        return {"version": SCHEMA_VERSION, "last_updated": "", "entries": {}}


def save_metadata(library_path: str | Path, data: dict) -> None:
    """Atomic write of the cache file to the share."""
    path = metadata_file_path(library_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data["last_updated"] = datetime.now().isoformat()
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(str(tmp), str(path))


def _entry_mtime(entry: dict) -> str:
    """Most recent _mtime_iso across an entry's files — our cache key."""
    best = ""
    for f in entry.get("files", []):
        m = f.get("_mtime_iso", "") or ""
        if m > best:
            best = m
    return best


def _extract_one(entry_id: str, title: str, text: str) -> dict:
    """
    Call the LLM once and parse its JSON. Returns a record dict on success
    or a minimal stub on failure so the cache always has an entry per
    processed study (prevents endless retry loops on a bad document).
    """
    if not text.strip():
        return {
            "entry_id":     entry_id,
            "product":      "",
            "model":        "",
            "endpoints":    [],
            "timepoints":   [],
            "extracted_at": datetime.now().isoformat(),
            "ok":           False,
            "error":        "no text",
        }

    trimmed = text[:EXTRACTION_TEXT_CAP]
    messages = [
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
        {"role": "user",   "content": (
            f"Study title: {title}\n\n"
            f"Study text:\n{trimmed}"
        )},
    ]
    try:
        resp = iliad_client.post_chat(
            messages, max_tokens=400, stream=False, timeout=60
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return {
            "entry_id":     entry_id,
            "product":      "",
            "model":        "",
            "endpoints":    [],
            "timepoints":   [],
            "extracted_at": datetime.now().isoformat(),
            "ok":           False,
            "error":        f"{type(e).__name__}: {e}",
        }

    raw = _extract_text_from_response(data)
    parsed = _parse_json_loose(raw)
    if not isinstance(parsed, dict):
        return {
            "entry_id":     entry_id,
            "product":      "",
            "model":        "",
            "endpoints":    [],
            "timepoints":   [],
            "extracted_at": datetime.now().isoformat(),
            "ok":           False,
            "error":        "unparseable response",
        }

    return {
        "entry_id":     entry_id,
        "product":      str(parsed.get("product", "") or "").strip(),
        "model":        str(parsed.get("model", "") or "").strip(),
        "endpoints":    [str(x).strip() for x in parsed.get("endpoints", []) if str(x).strip()],
        "timepoints":   [str(x).strip() for x in parsed.get("timepoints", []) if str(x).strip()],
        "extracted_at": datetime.now().isoformat(),
        "ok":           True,
        "error":        "",
    }


def _extract_text_from_response(data: dict) -> str:
    """Pick the reply text out of whichever response shape ILIAD returned."""
    if isinstance(data.get("content"), list):
        parts = [b.get("text", "") for b in data["content"]
                 if isinstance(b, dict) and b.get("type") == "text"]
        if parts:
            return "".join(parts).strip()
    if isinstance(data.get("completion"), dict):
        return str(data["completion"].get("content", "")).strip()
    if isinstance(data.get("choices"), list) and data["choices"]:
        return str(data["choices"][0].get("message", {}).get("content", "")).strip()
    if isinstance(data.get("message"), dict):
        return str(data["message"].get("content", "")).strip()
    if isinstance(data.get("message"), str):
        return data["message"].strip()
    return ""


def _parse_json_loose(text: str):
    """
    Try to parse a JSON object from `text`, tolerating markdown fences
    and a small amount of pre/post amble. Isolates the first top-level
    brace pair and parses that — the brace search naturally skips past
    ```json fences without special handling, which keeps the logic
    simple. Returns None if no object is recoverable.
    """
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except Exception:
        return None


def plan_extraction(
    library_index_entries: list[dict],
    cache: dict,
) -> tuple[list[dict], list[str]]:
    """
    Compare live index entries to the cache. Returns:
      - list of (entry, current_mtime) pairs that need re-extraction
      - list of cached entry_ids that no longer exist in the index
        (caller can prune them)
    """
    live_ids = {e.get("id", "") for e in library_index_entries if not e.get("deleted")}
    cached = cache.get("entries", {})

    to_extract: list[dict] = []
    for e in library_index_entries:
        if e.get("deleted"):
            continue
        eid = e.get("id", "")
        if not eid:
            continue
        current_mtime = _entry_mtime(e)
        prior = cached.get(eid)
        if (prior is None
            or prior.get("source_mtime", "") != current_mtime
            or not prior.get("ok", False)):
            to_extract.append({"entry": e, "mtime": current_mtime})

    orphan_ids = [eid for eid in cached.keys() if eid not in live_ids]
    return to_extract, orphan_ids


def run_extraction(
    library_path: str | Path,
    library_index_entries: list[dict],
    extract_text_fn: Callable[[str], str],
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict:
    """
    Incrementally refresh the AI metadata cache.

    Args:
      library_path: path to the Library/ folder.
      library_index_entries: the raw `entries` list from library_index.json.
      extract_text_fn: called with a filepath, returns extracted text.
        Supplied by the caller so this module stays free of heavy
        dependencies like pymupdf — the Streamlit app can pass in the
        cached variant from comparator.py.
      progress_callback: optional progress reporter (done, total, msg).

    Returns a summary dict: {extracted: N, skipped: K, orphaned: M, errors: E}.
    """
    cache = load_metadata(library_path)
    to_extract, orphan_ids = plan_extraction(library_index_entries, cache)

    # Prune removed entries first so the file doesn't grow forever.
    for oid in orphan_ids:
        cache["entries"].pop(oid, None)

    total = len(to_extract)
    skipped = len(library_index_entries) - total  # rough — includes deleted
    errors = 0

    if total == 0:
        save_metadata(library_path, cache)
        if progress_callback:
            progress_callback(0, 0, "Up to date")
        return {
            "extracted": 0,
            "skipped":   skipped,
            "orphaned":  len(orphan_ids),
            "errors":    0,
        }

    def _one(pair: dict) -> dict:
        entry = pair["entry"]
        mtime = pair["mtime"]
        title = entry.get("display_name") or entry.get("title", "")
        # Concatenate text from all files in the entry
        pieces: list[str] = []
        for f in entry.get("files", []):
            fp = f.get("filepath", "")
            if not fp:
                continue
            t = extract_text_fn(fp) or ""
            if t:
                pieces.append(f"[{Path(fp).name}]\n{t}")
        combined = "\n\n".join(pieces)
        result = _extract_one(entry.get("id", ""), title, combined)
        result["source_mtime"] = mtime
        return result

    done = 0
    with ThreadPoolExecutor(max_workers=EXTRACTION_WORKERS) as pool:
        futures = {pool.submit(_one, p): p for p in to_extract}
        for fut in as_completed(futures):
            rec = fut.result()
            eid = rec.get("entry_id", "")
            if eid:
                cache["entries"][eid] = rec
                if not rec.get("ok", False):
                    errors += 1
            done += 1
            if progress_callback:
                pair = futures[fut]
                title = (
                    pair["entry"].get("display_name")
                    or pair["entry"].get("title", "")
                )[:60]
                progress_callback(done, total, f"Extracting: {title}…")

    save_metadata(library_path, cache)
    return {
        "extracted": done,
        "skipped":   skipped,
        "orphaned":  len(orphan_ids),
        "errors":    errors,
    }


def load_entries_keyed(library_path: str | Path) -> dict:
    """Convenience: return {entry_id: metadata_dict} for downstream features."""
    return load_metadata(library_path).get("entries", {})
