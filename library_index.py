"""
library_index.py
Manages the library_index.json file that lives inside the Library folder.

The index stores one entry per document group (file or folder):
{
    "id":                unique stable key (folder name or first-file stem),
    "title":             display title,
    "auto_category":     category detected from filename/folder keywords,
    "manual_category":   admin override, or null if not overridden,
    "category":          resolved category (manual if set, else auto),
    "files": [
        {
            "filename": "...",
            "filepath": "...",
            "ext":      ".pdf",
            "size_kb":  42.1,
            "pages":    8,
            "modified": "14 Jan 2025",
            "ver_tuple": [0, 0],
            "ver_label": "",
        }, ...
    ],
    "preview":      "First paragraph text…",
    "modified":     "most recent modified date across files",
    "has_versions": false,
    "source":       "file" | "folder",
    "display_name": null,   # admin-set custom display name, overrides title
    "deleted":      false,  # soft-delete flag
    "last_indexed": "ISO timestamp"
}
"""

import json
import os
import threading
from datetime import datetime
from pathlib import Path

from config import LIBRARY_PATH, INDEX_FILENAME

INDEX_PATH = str(Path(LIBRARY_PATH) / INDEX_FILENAME)

_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Read / write
# ---------------------------------------------------------------------------

def load_index() -> dict[str, dict]:
    """Load the index from disk. Returns empty dict if not yet created."""
    try:
        with open(INDEX_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {entry["id"]: entry for entry in data.get("entries", [])}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_index(index: dict[str, dict]) -> None:
    """Persist the index to disk atomically."""
    with _lock:
        payload = {
            "version":      2,
            "last_updated": datetime.now().isoformat(),
            "entries":      list(index.values()),
        }
        tmp = INDEX_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, INDEX_PATH)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iso_modified(filepath: str) -> str:
    """Return ISO-format mtime string for comparison."""
    return datetime.fromtimestamp(Path(filepath).stat().st_mtime).isoformat()


def _needs_reindex(entry: dict) -> bool:
    """
    Return True if any file in the entry has been modified since last indexed,
    or if any file no longer exists (entry needs removal or update).
    """
    for f in entry.get("files", []):
        fp = Path(f["filepath"])
        if not fp.exists():
            return True
        if _iso_modified(f["filepath"]) != f.get("_mtime_iso", ""):
            return True
    return False


def _make_entry_id(source_path: Path) -> str:
    """Stable ID: folder name or file stem."""
    return source_path.name


# ---------------------------------------------------------------------------
# Full scan — runs on first launch or when triggered by admin
# ---------------------------------------------------------------------------

def full_scan(processor_fn) -> dict[str, dict]:
    """
    Scan the entire Library folder using processor_fn (imported from comparator
    to avoid circular imports) and build a fresh index.

    processor_fn signature:
        processor_fn(library_path: str) -> dict[str, list[dict]]
    where the return value is the categorised library as built by the app.

    Returns the new index dict.
    """
    lib_path = Path(LIBRARY_PATH)
    if not lib_path.exists():
        return {}

    # Use the app's own categorisation logic
    categorised = processor_fn(LIBRARY_PATH)

    index: dict[str, dict] = {}
    now = datetime.now().isoformat()

    for category, groups in categorised.items():
        for group in groups:
            entry_id = group.get("_id") or _make_entry_id_from_group(group)
            # Attach mtime_iso to each file for change detection
            files_with_mtime = []
            for f in group.get("files", []):
                fc = dict(f)
                try:
                    fc["_mtime_iso"] = _iso_modified(f["filepath"])
                except OSError:
                    fc["_mtime_iso"] = ""
                files_with_mtime.append(fc)

            existing = index.get(entry_id, {})
            index[entry_id] = {
                "id":              entry_id,
                "title":           group.get("title", ""),
                "auto_category":   category,
                "manual_category": existing.get("manual_category"),
                "category":        existing.get("manual_category") or category,
                "files":           files_with_mtime,
                "preview":         group.get("preview", ""),
                "modified":        group.get("modified", ""),
                "has_versions":    group.get("has_versions", False),
                "source":          group.get("source", "file"),
                "display_name":    existing.get("display_name"),
                "deleted":         existing.get("deleted", False),
                "last_indexed":    now,
            }

    save_index(index)
    return index


def _make_entry_id_from_group(group: dict) -> str:
    files = group.get("files", [])
    if files:
        return Path(files[0]["filepath"]).parent.name + "_" + Path(files[0]["filename"]).stem
    return group.get("title", "unknown").replace(" ", "_")


# ---------------------------------------------------------------------------
# Incremental scan — checks only for new/changed/deleted files
# ---------------------------------------------------------------------------

def incremental_scan(existing_index: dict[str, dict], processor_fn) -> dict[str, dict]:
    """
    Fast update: only re-processes entries whose files have changed,
    and adds any new files not yet in the index.
    """
    lib_path = Path(LIBRARY_PATH)
    if not lib_path.exists():
        return existing_index

    # Find all current items on disk
    current_items = set()
    for item in lib_path.iterdir():
        if item.name == INDEX_FILENAME:
            continue
        if item.is_file() or item.is_dir():
            current_items.add(item.name)

    # Check for deletions (items in index no longer on disk)
    index = dict(existing_index)
    for entry_id, entry in list(index.items()):
        source_names = {Path(f["filepath"]).parent.name
                        if entry["source"] == "folder"
                        else Path(f["filepath"]).name
                        for f in entry["files"]}
        if not source_names.intersection(current_items):
            # All source files gone — mark deleted
            index[entry_id]["deleted"] = True

    # Check for new or changed items
    indexed_sources = set()
    for entry in index.values():
        for f in entry.get("files", []):
            indexed_sources.add(Path(f["filepath"]).name)
            indexed_sources.add(Path(f["filepath"]).parent.name)

    new_items = current_items - indexed_sources - {INDEX_FILENAME}
    stale_entries = [eid for eid, e in index.items() if _needs_reindex(e)]

    if not new_items and not stale_entries:
        return index  # Nothing changed

    # Re-run full scan only if there are changes
    fresh = full_scan(processor_fn)

    # Preserve manual overrides and display names from existing index
    for entry_id, fresh_entry in fresh.items():
        if entry_id in index:
            fresh_entry["manual_category"] = index[entry_id].get("manual_category")
            fresh_entry["display_name"]    = index[entry_id].get("display_name")
            fresh_entry["category"]        = (
                fresh_entry["manual_category"] or fresh_entry["auto_category"]
            )
            fresh_entry["deleted"] = index[entry_id].get("deleted", False)

    save_index(fresh)
    return fresh


# ---------------------------------------------------------------------------
# Admin operations
# ---------------------------------------------------------------------------

def override_category(index: dict, entry_id: str, new_category: str) -> dict:
    """Set a manual category override for an entry."""
    if entry_id in index:
        index[entry_id]["manual_category"] = new_category
        index[entry_id]["category"]        = new_category
        save_index(index)
    return index


def rename_entry(index: dict, entry_id: str, new_name: str) -> dict:
    """Set a custom display name for an entry."""
    if entry_id in index:
        index[entry_id]["display_name"] = new_name.strip() or None
        save_index(index)
    return index


def delete_entry(index: dict, entry_id: str) -> dict:
    """Soft-delete an entry from the index (file stays on disk)."""
    if entry_id in index:
        index[entry_id]["deleted"] = True
        save_index(index)
    return index


def restore_entry(index: dict, entry_id: str) -> dict:
    """Restore a soft-deleted entry."""
    if entry_id in index:
        index[entry_id]["deleted"] = False
        save_index(index)
    return index


def reset_category(index: dict, entry_id: str) -> dict:
    """Remove manual override, revert to auto-detected category."""
    if entry_id in index:
        index[entry_id]["manual_category"] = None
        index[entry_id]["category"]        = index[entry_id]["auto_category"]
        save_index(index)
    return index
