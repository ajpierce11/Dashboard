"""
vector_store.py
Semantic document retrieval using ILIAD embeddings + numpy vector search.

No external vector DB required — embeddings are stored as a numpy array
in a .npz file alongside the library index, loaded into memory on first use.

Usage:
    from vector_store import VectorStore
    vs = VectorStore(library_path, api_key)
    vs.build()                          # index all documents (run once)
    results = vs.search("my question", top_k=8)
    # returns list of {"title": ..., "text": ..., "category": ...}
"""

import json
from pathlib import Path

import numpy as np

import iliad_client
from title_utils import base_title

EMBED_DIM   = 1536
CHUNK_SIZE  = 1200    # characters per chunk
CHUNK_OVERLAP = 150  # overlap between chunks to preserve context
BATCH_SIZE  = 50     # documents per embedding API call (well under the 2048 limit)

VECTOR_FILE = "library_vectors.npz"   # saved in the Library folder


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks at sentence boundaries where possible."""
    if not text:
        return []
    # Hard safety cap — never process more than 8000 chars
    if len(text) < 50:
        return [text]

    chunks = []
    start  = 0
    max_chunks = 80  # up to 80 chunks for a 30-page document

    while start < len(text) and len(chunks) < max_chunks:
        end = min(start + size, len(text))
        if end < len(text):
            for sep in (". ", ".\n", "! ", "? ", "\n\n", "\n"):
                pos = text.rfind(sep, start + size // 2, end)
                if pos != -1:
                    end = pos + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
        if start >= len(text):
            break

    return chunks


# ---------------------------------------------------------------------------
# Embedding API
# ---------------------------------------------------------------------------

def _embed_batch(texts: list[str], api_key: str) -> list[list[float]]:
    """Thin wrapper around iliad_client.embed kept for backwards-compat callers."""
    return iliad_client.embed(texts, api_key=api_key)


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

class VectorStore:
    """
    In-memory vector store backed by a .npz file on disk.

    Each entry in the store is a text chunk from a library document.
    Metadata (title, category, source entry_id) is stored alongside.
    Similarity search uses cosine similarity computed with numpy.
    """

    def __init__(self, library_path: str, api_key: str):
        self.library_path  = Path(library_path)
        self.api_key       = api_key
        self.vector_file   = self.library_path / VECTOR_FILE
        self.index_file    = self.library_path / "library_index.json"

        # Loaded into memory on first use
        self._vectors: np.ndarray | None = None   # shape (N, EMBED_DIM)
        self._metadata: list[dict]       = []     # parallel list of chunk metadata

    # ── Persistence ──────────────────────────────────────────────────────────

    def _save(self) -> None:
        """Save vectors and metadata to disk."""
        if self._vectors is None or len(self._vectors) == 0:
            return
        np.savez_compressed(
            str(self.vector_file),
            vectors=self._vectors,
            metadata=np.array(
                [json.dumps(m) for m in self._metadata], dtype=object
            ),
        )

    def _load(self) -> bool:
        """Load vectors and metadata from disk. Returns True if successful."""
        if not self.vector_file.exists():
            return False
        try:
            data = np.load(str(self.vector_file), allow_pickle=True)
            self._vectors  = data["vectors"]
            self._metadata = [json.loads(m) for m in data["metadata"]]
            return True
        except Exception:
            return False

    def is_built(self) -> bool:
        return self.vector_file.exists()

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self, progress_callback=None) -> dict:
        """
        Read the library index, chunk and embed documents one at a time,
        accumulating vectors incrementally to avoid loading everything into
        memory at once.

        progress_callback(done, total, message) — optional UI feedback.
        Returns {"chunks": N, "documents": M, "errors": K}
        """
        if not self.index_file.exists():
            return {"error": "Library index not found. Build the library index first."}

        with open(self.index_file, "r", encoding="utf-8") as f:
            index_data = json.load(f)

        entries = [e for e in index_data.get("entries", []) if not e.get("deleted")]
        if not entries:
            return {"error": "No documents in library index."}

        total_docs    = len(entries)
        all_metadata: list[dict] = []
        errors        = 0
        total_chunks  = 0
        CHECKPOINT    = 50   # save to disk every N documents to cap memory use

        # Start fresh
        if self.vector_file.exists():
            self.vector_file.unlink()
        self._vectors  = None
        self._metadata = []

        batch_vectors:  list[np.ndarray] = []
        batch_metadata: list[dict]       = []

        def _flush():
            nonlocal batch_vectors, batch_metadata
            if not batch_vectors:
                return
            new_arr = np.stack(batch_vectors, axis=0)
            if self._vectors is None:
                self._vectors = new_arr
            else:
                self._vectors = np.concatenate([self._vectors, new_arr], axis=0)
            self._metadata.extend(batch_metadata)
            self._save()
            del new_arr
            batch_vectors  = []
            batch_metadata = []

        for doc_idx, entry in enumerate(entries):
            title    = entry.get("display_name") or entry.get("title", "")
            category = entry.get("category", "")
            text_to_embed = entry.get("full_text", "") or entry.get("preview", "")
            entry_id      = entry.get("id", "")

            if not text_to_embed:
                continue

            if progress_callback:
                progress_callback(
                    doc_idx, total_docs,
                    f"Document {doc_idx + 1}/{total_docs}: {title[:55]}..."
                )

            # Hard cap preview to prevent memory errors on large documents
            chunks = _chunk_text(text_to_embed)
            if not chunks:
                continue

            try:
                vecs = _embed_batch(chunks, self.api_key)
                if not vecs:
                    print(f"  WARNING: empty response for: {title[:60]}")
                    errors += len(chunks)
                    continue
                for i, (chunk, vec) in enumerate(zip(chunks, vecs)):
                    batch_vectors.append(np.array(vec, dtype=np.float32))
                    batch_metadata.append({
                        "title":    title,
                        "category": category,
                        "entry_id": entry_id,
                        "chunk_i":  i,
                        "text":     chunk,
                        "modified":  entry.get("modified", ""),
                        "mtime_iso": max(
                            (f.get("_mtime_iso", "") for f in entry.get("files", [])),
                            default=""
                        ),
                    })
                total_chunks += len(chunks)
            except Exception as e:
                print(f"  ERROR embedding {title[:60]}: {e}")
                errors += len(chunks)

            if (doc_idx + 1) % CHECKPOINT == 0:
                if progress_callback:
                    progress_callback(doc_idx + 1, total_docs, "Saving checkpoint...")
                _flush()

        _flush()

        if self._vectors is None:
            return {"error": "No text content could be embedded."}

        if progress_callback:
            progress_callback(total_docs, total_docs, "Done.")

        return {
            "chunks":    total_chunks,
            "documents": total_docs,
            "errors":    errors,
        }

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, question: str, top_k: int = 8) -> list[dict]:
        """
        Embed the question and return the top_k most semantically similar
        document chunks. Deduplicates by document (one result per entry_id).

        Returns list of {"title", "category", "text", "score"}.
        """
        if self._vectors is None:
            if not self._load():
                return []

        if self._vectors is None or len(self._vectors) == 0:
            return []

        # Embed the question
        try:
            q_vecs = _embed_batch([question], self.api_key)
            if not q_vecs:
                return []
            q_vec = np.array(q_vecs[0], dtype=np.float32)
        except Exception:
            return []

        # Cosine similarity: dot product of normalised vectors
        norms = np.linalg.norm(self._vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-9, norms)
        normed = self._vectors / norms

        q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-9)
        scores = normed @ q_norm   # shape (N,)

        # Sort descending — take a large candidate pool so we don't miss
        # lower-ranked chunks from revision versions of top documents
        ranked = np.argsort(scores)[::-1]

        from collections import defaultdict as _dd

        def _base(title: str) -> str:
            return base_title(title, include_study_prefix=True)

        # Step 1: Find top-k unique base titles from highest-scoring chunks
        # Use a minimum score threshold so documents mentioned only in body
        # text (not titles) still get included
        MIN_SCORE = 0.10  # low threshold — better to retrieve too much than miss relevant docs
        top_bases: list[str] = []
        seen_bases: set[str] = set()
        for idx in ranked:
            score_val = float(scores[idx])
            # Stop if score drops below threshold AND we have enough results
            if score_val < MIN_SCORE and len(top_bases) >= 3:
                break
            base = _base(self._metadata[idx]["title"])
            if base not in seen_bases:
                seen_bases.add(base)
                top_bases.append(base)
            if len(top_bases) >= top_k:
                break

        # Build (entry_id, chunk_i) -> score_index lookup once so chunk-level
        # scoring inside the merge loop is O(1) instead of O(N) per chunk.
        chunk_idx: dict[tuple, int] = {
            (m.get("entry_id"), m.get("chunk_i")): i
            for i, m in enumerate(self._metadata)
        }

        # Step 2: For each top base title, gather ALL chunks from ALL versions
        # by scanning the entire metadata list — this ensures Rev3 text is
        # included even if its chunks scored lower than _final's chunks
        base_chunks: dict[str, list] = {b: [] for b in top_bases}
        base_scores: dict[str, float] = {b: 0.0 for b in top_bases}

        for idx in range(len(self._metadata)):
            meta = self._metadata[idx]
            base = _base(meta["title"])
            if base in base_chunks:
                base_chunks[base].append(meta)
                base_scores[base] = max(base_scores[base], float(scores[idx]))

        def _chunk_score(m: dict) -> float:
            i = chunk_idx.get((m.get("entry_id"), m.get("chunk_i")))
            return float(scores[i]) if i is not None else 0.0

        # Step 3: Build results — merge all version texts per base
        results: list[dict] = []
        for base in top_bases:
            chunks = base_chunks[base]
            if not chunks:
                continue

            # Group by entry_id (each unique file), then take the best chunk per file
            by_entry: dict = _dd(list)
            for meta in chunks:
                by_entry[meta["entry_id"]].append(meta)

            # Merge one representative chunk per version, labelled by title
            seen_titles: set[str] = set()
            version_texts: list[str] = []
            for entry_id, entry_chunks in by_entry.items():
                # Score each chunk individually, pick top 20, re-sort by position
                # so the most relevant sections are sent, not just the intro
                top_chunks = sorted(entry_chunks, key=_chunk_score, reverse=True)[:20]
                top_chunks = sorted(top_chunks, key=lambda m: m.get("chunk_i", 0))
                title = top_chunks[0]["title"]
                if title not in seen_titles:
                    seen_titles.add(title)
                    combined = "\n".join(c["text"] for c in top_chunks)
                    version_texts.append(f"[{title}]:\n{combined}")

            merged = "\n\n".join(version_texts)
            best_meta = max(chunks, key=_chunk_score)

            results.append({
                "title":    best_meta["title"],
                "category": best_meta["category"],
                "text":     merged[:40000],
                "score":    base_scores[base],
            })

        return results

    def needs_rebuild(self) -> bool:
        """
        Return True if the vector store is older than the library index,
        meaning new documents have been added since the last build.
        """
        if not self.vector_file.exists():
            return True
        if not self.index_file.exists():
            return False
        return (
            self.vector_file.stat().st_mtime < self.index_file.stat().st_mtime
        )

    def update(self, progress_callback=None) -> dict:
        """
        Incremental update — only embeds documents not already in the vector store.

        Compares entry_ids in the current index against entry_ids already in the
        vector store. Only new entries are processed and embedded, then appended
        to the existing vectors. Deleted entries are pruned.

        For 1-2 new files this takes seconds rather than minutes.
        Returns {"new_chunks": N, "new_documents": M, "removed": K, "errors": E}
        """
        if not self.index_file.exists():
            return {"error": "Library index not found."}

        # Load existing vectors if not already in memory
        if self._vectors is None:
            if not self._load():
                # No existing store — fall back to full build
                return self.build(progress_callback=progress_callback)

        with open(self.index_file, "r", encoding="utf-8") as f:
            index_data = json.load(f)

        entries = [e for e in index_data.get("entries", []) if not e.get("deleted")]
        current_ids = {e["id"] for e in entries}

        # Find which entry_ids are already in the vector store
        existing_ids = {m["entry_id"] for m in self._metadata}

        # New entries to embed
        new_entries = [e for e in entries if e["id"] not in existing_ids]

        # Entries to remove (deleted from index)
        removed_ids = existing_ids - current_ids

        errors = 0
        new_chunks_count = 0

        # ── Remove deleted entries ────────────────────────────────────────────
        if removed_ids:
            keep_mask = np.array(
                [m["entry_id"] not in removed_ids for m in self._metadata]
            )
            self._vectors  = self._vectors[keep_mask]
            self._metadata = [m for m, k in zip(self._metadata, keep_mask) if k]

        # ── Embed new entries ─────────────────────────────────────────────────
        if new_entries:
            new_chunks:   list[str]  = []
            new_meta:     list[dict] = []

            for entry in new_entries:
                title    = entry.get("display_name") or entry.get("title", "")
                category = entry.get("category", "")
                text_to_embed = entry.get("full_text", "") or entry.get("preview", "")
                entry_id      = entry.get("id", "")
                if not text_to_embed:
                    continue
                for i, chunk in enumerate(_chunk_text(text_to_embed)):
                    new_chunks.append(chunk)
                    new_meta.append({
                        "title":    title,
                        "category": category,
                        "entry_id": entry_id,
                        "chunk_i":  i,
                        "text":     chunk,
                        "modified":  entry.get("modified", ""),
                        "mtime_iso": max(
                            (f.get("_mtime_iso", "") for f in entry.get("files", [])),
                            default=""
                        ),
                    })

            total = len(new_chunks)
            new_vectors: list[list[float]] = []
            done = 0

            for batch_start in range(0, total, BATCH_SIZE):
                batch = new_chunks[batch_start:batch_start + BATCH_SIZE]
                if progress_callback:
                    progress_callback(
                        done, total,
                        f"Embedding {batch_start+1}–{min(batch_start+len(batch), total)} "
                        f"of {total} new chunks…"
                    )
                try:
                    vecs = _embed_batch(batch, self.api_key)
                    new_vectors.extend(vecs)
                    done += len(batch)
                except Exception:
                    new_vectors.extend([[0.0] * EMBED_DIM] * len(batch))
                    errors += len(batch)
                    done += len(batch)

            if new_vectors:
                new_arr = np.array(new_vectors, dtype=np.float32)
                self._vectors  = np.concatenate([self._vectors, new_arr], axis=0)
                self._metadata = self._metadata + new_meta
                new_chunks_count = len(new_chunks)

        if progress_callback:
            progress_callback(1, 1, "Saving…")

        self._save()

        return {
            "new_chunks":    new_chunks_count,
            "new_documents": len(new_entries),
            "removed":       len(removed_ids),
            "errors":        errors,
        }