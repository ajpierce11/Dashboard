"""
config.py
Central configuration for the AbbVie Testing Dashboard.

Paths are resolved in this order:
  1. Environment variable (if set) — TESTING_FILE_PATH, LIBRARY_PATH
  2. Default relative to this file's directory (the repo root)

This lets every teammate/CML session run the app without editing source.
Set overrides via a local .env file (see .env.example) or the shell.
"""

import os
from pathlib import Path

# Load .env if python-dotenv is installed (safe no-op otherwise)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Base directory — the folder containing this file (repo root)
# ---------------------------------------------------------------------------
_BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
FILE_PATH = os.environ.get(
    "TESTING_FILE_PATH",
    str(_BASE_DIR / "testing fpt files.xlsx"),
)

LIBRARY_PATH = os.environ.get(
    "LIBRARY_PATH",
    str(_BASE_DIR / "Library"),
)

# Name of the index file stored inside the Library folder
INDEX_FILENAME = "library_index.json"
