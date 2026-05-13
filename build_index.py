"""
build_index.py
Builds the vector search index by reading documents directly from disk.
Does NOT rely on the library_index.json having full_text populated.
"""
import os, sys, time, json
from pathlib import Path

print("build_index.py starting...")

sys.path.insert(0, str(Path(__file__).parent))
from config import LIBRARY_PATH

API_KEY = os.environ.get("ILIAD_API_KEY", "")

print(f"API key set: {bool(API_KEY)}")
print(f"Library exists: {Path(LIBRARY_PATH).exists()}")

if not API_KEY:
    print("ERROR: ILIAD_API_KEY not set"); input(); sys.exit(1)
if not Path(LIBRARY_PATH).exists():
    print("ERROR: Library folder not found"); input(); sys.exit(1)

index_path = Path(LIBRARY_PATH) / "library_index.json"
if not index_path.exists():
    print("ERROR: library_index.json not found. Open the app first."); input(); sys.exit(1)

# ── Read index to get document list and metadata ──────────────────────────────
print("\nReading library index...")
with open(index_path, "r", encoding="utf-8") as f:
    index_data = json.load(f)

entries = [e for e in index_data.get("entries", []) if not e.get("deleted")]
print(f"  Found {len(entries)} documents")

# ── Extract full text directly from files (bypasses index full_text field) ───
print("\nStep 1: Extracting full text from documents...")

from doc_text import extract_text, SUPPORTED_EXTS as SUPPORTED


# Build a list of (entry_id, title, category, text) for embedding
docs_to_embed = []

for entry in entries:
    title    = entry.get("display_name") or entry.get("title", "")
    category = entry.get("category", "")
    entry_id = entry.get("id", "")
    files    = entry.get("files", [])
    
    # Extract text from all files in this entry
    all_texts = []
    for f in files:
        fp  = f.get("filepath", "")
        ext = f.get("ext", "").lower()
        if ext in SUPPORTED and fp:
            text = extract_text(fp)
            if text:
                all_texts.append(f"[{Path(fp).name}]\n{text}")
    
    combined = "\n\n".join(all_texts)
    
    if combined:
        docs_to_embed.append({
            "entry_id": entry_id,
            "title":    title,
            "category": category,
            "text":     combined,
        })
    else:
        # No text extracted — use title as minimal fallback
        docs_to_embed.append({
            "entry_id": entry_id,
            "title":    title,
            "category": category,
            "text":     title,
        })

print(f"  {sum(1 for d in docs_to_embed if len(d['text']) > len(d['title']))} documents with text extracted")
print(f"  {sum(1 for d in docs_to_embed if len(d['text']) == len(d['title']))} documents with title only (no text)")

# ── Build vector store ────────────────────────────────────────────────────────
print("\nStep 2: Building vector index...")

from vector_store import _chunk_text, _embed_batch
import numpy as np

CHECKPOINT = 5
vector_file = Path(LIBRARY_PATH) / "library_vectors.npz"

# Load existing vectors for incremental update
existing_vectors  = None
existing_metadata = []
existing_ids: set = set()

if vector_file.exists():
    print("  Existing index found — only embedding new documents...")
    try:
        data = np.load(str(vector_file), allow_pickle=True)
        existing_vectors  = data["vectors"]
        existing_metadata = [json.loads(m) for m in data["metadata"]]
        existing_ids      = {m["entry_id"] for m in existing_metadata}
        print(f"  Already indexed: {len(existing_ids)} documents ({len(existing_metadata)} chunks)")
    except Exception as e:
        print(f"  Could not load existing index ({e}) — full rebuild")
else:
    print("  No existing index — full build...")

# Only process new documents
new_docs = [d for d in docs_to_embed if d["entry_id"] not in existing_ids]
print(f"  New documents to embed: {len(new_docs)}  (skipping {len(docs_to_embed)-len(new_docs)} already indexed)")

if not new_docs:
    print("\n  Nothing to do — all documents already indexed!")
    print(f"  Total chunks in store: {len(existing_metadata)}")
    input("\nFinished. Press Enter to exit.")
    import sys; sys.exit(0)

docs_to_embed = new_docs

all_new_vectors:  list[np.ndarray] = []
all_new_metadata: list[dict]       = []
batch_v: list[np.ndarray] = []
batch_m: list[dict]       = []
errors       = 0
total_chunks = 0
start        = time.time()

def flush():
    global batch_v, batch_m, all_new_vectors, all_new_metadata
    if not batch_v:
        return
    arr = np.stack(batch_v, axis=0)
    all_new_vectors.append(arr)
    all_new_metadata.extend(batch_m)
    # Merge new with existing and save
    new_arr = np.concatenate(all_new_vectors, axis=0) if len(all_new_vectors) > 1 else all_new_vectors[0]
    if existing_vectors is not None:
        save_v = np.concatenate([existing_vectors, new_arr], axis=0)
        save_m = existing_metadata + all_new_metadata
    else:
        save_v = new_arr
        save_m = all_new_metadata

    # Stamp the library index's last_updated into the archive so that
    # VectorStore.needs_rebuild() can compare logical versions rather
    # than filesystem mtimes. Without this, the "consider rebuilding"
    # warning reappears permanently after every CLI build.
    index_stamp = ""
    try:
        with open(index_path, "r", encoding="utf-8") as _f:
            index_stamp = json.load(_f).get("last_updated", "")
    except Exception:
        pass

    # Atomic write: .tmp + os.replace. A second writer or a crash mid-save
    # can no longer leave a partial .npz that every reader fails to load.
    tmp_file = vector_file.with_suffix(vector_file.suffix + ".tmp")
    np.savez_compressed(
        str(tmp_file),
        vectors=save_v,
        metadata=np.array([json.dumps(m) for m in save_m], dtype=object),
        index_stamp=np.array([index_stamp], dtype=object),
    )
    os.replace(str(tmp_file), str(vector_file))
    batch_v = []
    batch_m = []

# Entry IDs that have historically caused the ILIAD embedding endpoint to
# stall past the per-batch timeout. Populated manually after investigation
# — add an entry here only after confirming the document reliably exhausts
# the budget. Re-test periodically and remove entries that start working
# again (likely after an ILIAD update).
SKIP_ENTRY_IDS: set = {
    "1876-D46-055 - NOA Hydration Lift+TI_TechnicalReport_final_docx",
    "1876-D46-055 - NOA Hydration Lift+TI_TechnicalReport_final_pdf",
    "1876-D46-055 - NOA Hydration TI_TechnicalReport_avp-final_docx",
    "1876-D46-055 NOA Hydration Lift+TI_protocol_pdf",
    "2055-D01-065 - Prolonged NOA Hydration TI_TechnicalReport_FINAL-LN_docx",
    "2055-D01-065 - Prolonged NOA Hydration TI_TechnicalReport_Final_signed_pdf",
    "2055-D01-065 - Prolonged NOA Hydration TI_TechnicalReport_signed_pdf",
    "20220310-20220310_2055-D01-065_Prolonged NOA Hydration Protocol - signed_pdf",
    "2055-D01-065_Prolonged NOA Hydration Protocol_docx",
}

# Per-document overall budget. The HTTP request timeout inside
# iliad_client.embed protects each batch; this watchdog catches the
# pathological case where many batches each take close to the max.
DOC_TIMEOUT = 120

for di, doc in enumerate(docs_to_embed):
    if doc["entry_id"] in SKIP_ENTRY_IDS:
        print(f"  [SKIP] {doc['title'][:60]}")
        continue

    elapsed = time.time() - start
    pct = int(di / len(docs_to_embed) * 100)
    print(f"  [{pct:3d}%] {di+1}/{len(docs_to_embed)}: {doc['title'][:60]}...  ({elapsed:.0f}s)")

    chunks = _chunk_text(doc["text"])
    if not chunks:
        continue

    doc_start = time.time()

    # Send chunks in smaller batches of 10 to avoid ILIAD timeouts on
    # large docs. iliad_client.embed sets a per-HTTP-request timeout so a
    # single hung batch can't stall forever — no need for a thread watchdog.
    EMBED_BATCH = 10
    for chunk_start in range(0, len(chunks), EMBED_BATCH):
        if time.time() - doc_start > DOC_TIMEOUT:
            print(f"    ! Per-doc timeout — skipping remaining chunks")
            errors += len(chunks) - chunk_start
            break
        chunk_batch = chunks[chunk_start:chunk_start + EMBED_BATCH]
        try:
            vecs = _embed_batch(chunk_batch, API_KEY)
            if not vecs:
                print(f"    ! Empty embedding response for batch")
                errors += len(chunk_batch)
                continue
            for i, (chunk, vec) in enumerate(zip(chunk_batch, vecs)):
                batch_v.append(np.array(vec, dtype=np.float32))
                batch_m.append({
                    "title":    doc["title"],
                    "category": doc["category"],
                    "entry_id": doc["entry_id"],
                    "chunk_i":  chunk_start + i,
                    "text":     chunk,
                })
            total_chunks += len(chunk_batch)
        except Exception as e:
            print(f"    ! Error on batch: {e}")
            errors += len(chunk_batch)
    
    if (di + 1) % CHECKPOINT == 0:
        print(f"  Checkpoint save ({len(all_new_metadata) + len(batch_m)} new chunks so far)...")
        flush()

flush()

elapsed = time.time() - start
total_in_store = len(existing_metadata) + total_chunks
print(f"\n{'='*60}")
print(f"  Done in {elapsed:.0f}s")
print(f"  New documents embedded: {len(docs_to_embed)}")
print(f"  New chunks added: {total_chunks}")
print(f"  Total chunks in store: {total_in_store}")
if errors:
    print(f"  Errors: {errors}")
print(f"  Saved to: {vector_file}")
print(f"{'='*60}")

input("\nFinished. Press Enter to exit.")