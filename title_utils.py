"""
title_utils.py
Shared helpers for normalising document titles and filename stems.

Before this module existed, four different files each had their own copy of
"strip the Rev / final / signed / (signed) / v2 suffix". They had drifted
apart — the vector-store version additionally prefixed a study-number key,
while the library-scan version kept type words like "protocol" / "report".
Consolidating here so any future rule changes happen in one place.
"""

import re
from pathlib import Path

_STUDY_NUM_RE = re.compile(
    r"\b([A-Za-z0-9]+-[A-Za-z0-9]+-[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*)\b"
)

_VERSION_SUFFIX_PATTERNS = (
    r"[\s_\-]*(revision|rev)[\s_\-]*\d*",
    r"[\s_\-]*v\d+(\.\d+)?",
    r"[\s_\-]*(final|draft|clean|signed|approved|amended|updated?)",
    r"[\s_\-]*\(.*?\)",
)


def strip_version_suffix(title: str) -> str:
    """
    Remove trailing version/status markers repeatedly to handle combinations
    like "Protocol_final-signed (v2)".
    """
    t = title.strip()
    for _ in range(5):
        prev = t
        for pattern in _VERSION_SUFFIX_PATTERNS:
            t = re.sub(pattern + r"[\s_\-]*$", "", t, flags=re.IGNORECASE).strip()
        if t == prev:
            break
    return t


def base_title(title: str, include_study_prefix: bool = False) -> str:
    """
    Normalise a document title for grouping. All callers collapse different
    revisions of the same document into a single bucket using this.

    include_study_prefix=True prepends the extracted study number (e.g.
    "1745-D76-054|...") so two unrelated studies whose tails collide after
    suffix stripping still hash to different keys. The vector-store search
    path uses this; keyword retrieval does not.
    """
    stripped = strip_version_suffix(title).lower()
    if include_study_prefix:
        m = _STUDY_NUM_RE.search(title)
        prefix = m.group(1).lower() if m else ""
        return f"{prefix}|{stripped}"
    return stripped


def clean_stem(stem: str) -> str:
    """
    Normalise a filename stem for similarity comparison in the library
    grouping pass. Unlike base_title this keeps type words ("protocol",
    "report") so "AB-001 Protocol" and "AB-001 Report" remain distinct.
    """
    s = stem.lower()
    s = re.sub(r"[\s_\-]*(revision|rev)[\s_\-]*\d*[\s_\-]*$", "", s)
    s = re.sub(r"[\s_\-]*v\d+(\.\d+)?[\s_\-]*$", "", s)
    s = re.sub(
        r"[\s_\-]*(final|draft|clean|signed|approved|amended|updated?)[\s_\-]*$",
        "", s,
    )
    s = re.sub(r"[\s_\-]+", " ", s).strip()
    return s


def display_title_from_filename(filename: str) -> str:
    """Title-case a filename after stripping version suffixes and separators."""
    base = strip_version_suffix(Path(filename).stem)
    return base.replace("_", " ").replace("-", " ").strip()
