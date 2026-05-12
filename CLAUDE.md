# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the apps

This repo contains two independent applications.

**1. Testing Dashboard (top-level) — Streamlit**

```bash
streamlit run comparator.py
```

Requires env var `ILIAD_API_KEY` (AbbVie's internal LLM gateway). Without it, the AI Assistant tab shows a warning but the other tabs still work. Paths to the Excel workbook and Library folder live in `config.py` — edit there if network paths change.

**2. Protocol → Report Converter (`protocol_to_report/`) — Flask**

```bash
cd protocol_to_report
pip install -r requirements.txt
python -m spacy download en_core_web_md
python app.py   # http://localhost:5000
```

Fully offline; no API calls. Uses spaCy + pyinflect for tense conversion on .docx files.

## Vector index maintenance

The AI Assistant depends on `Library/library_index.json` (metadata) and `Library/library_vectors.npz` (embeddings).

- `library_index.json` is built/updated by `comparator.py` itself on launch (see `_build_index_from_scan` and `incremental_scan` in `library_index.py`). Admin panel in the Library tab can force a rebuild.
- `library_vectors.npz` is built separately. Use either the "Build index" button in the AI Assistant tab (calls `VectorStore.build`) or run the standalone CLI:

```bash
python build_index.py   # incremental — only embeds new docs
```

`build_index.py` reads text directly from disk rather than trusting the `full_text` field on the index, batches chunks in groups of 10, has a hard 25 s per-batch thread timeout, and maintains a hard-coded `SKIP_ENTRY_IDS` set for documents that hang the ILIAD embedding endpoint. When adding new docs, prefer `build_index.py` over a full rebuild.

Debug helpers:

```bash
python check_index.py    # report on library_index.json contents
python check_vectors.py  # report on library_vectors.npz contents
```

## Architecture

### Testing Dashboard — three tabs in one Streamlit app

`comparator.py` (~2500 lines) is the single entry point. `main()` sets up three tabs:

1. **Product Comparator** — Reads `testing fpt files.xlsx` via `load_data()`. The sheet has a fixed column layout (0: Product, 1–6: means at each timepoint, 7–12: std devs, 14: reference, 15–19: static properties). `load_data` melts into long form and returns four objects: long df, timepoint list, product→reference map, static properties df. Downstream chart/stats/export functions all consume these.
2. **Document Library** — File browser + admin panel over the `Library/` folder. Groups multi-file studies together (e.g. protocol + report under one "study" card), detects versions from filenames, and classifies each entry into a category (`classify_document` + `_reclassify_ambiguous`).
3. **AI Assistant** — Chatbot backed by ILIAD `gpt-4o-mini` with retrieval from two parallel sources: (a) keyword scoring in `_retrieve_library_context` (study-number + keyword heuristics over the JSON index), (b) semantic search via `VectorStore.search`. Context from both is concatenated into the system prompt.

### Library index / vector store split

Keep these two layers separate in your head:

- **`library_index.py`** — metadata only (titles, file paths, categories, preview text). Persists manual overrides (`manual_category`, `display_name`, `deleted`) across rebuilds. `incremental_scan` checks file mtimes and only re-processes changed entries.
- **`vector_store.py`** — embedding layer. Chunks text (1200 chars, 150 overlap), calls ILIAD `text-embedding-3-small`, stores as numpy `.npz`. `search()` deduplicates results by "base title" (strips version suffixes like `_final`, `Rev2`, `(signed)` repeatedly up to 5 passes) so different revisions of the same study collapse into one result.

Both layers must be rebuilt in order — `library_index.json` first, then `library_vectors.npz`.

### ILIAD integration

All LLM traffic goes through AbbVie's internal gateway at `iliad-emerging-api.abbvienet.com` with `verify=False` (corporate proxy uses a self-signed cert). Two endpoints:

- Chat: `/api/v1/chat/gpt-4o-mini` — defined as `ILIAD_URL` in `comparator.py`, called via `_call_iliad`. Response shape varies (OpenAI-style `choices`, ILIAD-style `completion`/`message`); `_call_iliad` tries each.
- Embeddings: `/api/v1/embed/text-embedding-3-small` — in `vector_store.py` as `EMBED_URL`, called via `_embed_batch` with exponential-backoff retries.

Auth header is `x-api-key: $ILIAD_API_KEY` (not Bearer). This is unrelated to the `ANTHROPIC_*` vars in `settings.json` — those configure the Claude Code harness itself.

### Duplicate code in `Dashboard App/`

The `Dashboard App/` subdirectory contains an older, near-identical copy of `comparator.py`, `build_index.py`, `vector_store.py`, etc. Top-level files are the source of truth (newer mtimes, more features — e.g. the top-level `vector_store.py` has the `update()` incremental method that the copy lacks). Make edits to the top-level files; do not assume the copy tracks them.

### `old/` directory

Contains historical backups (Accruals, Competitor Requests, etc.) unrelated to the current dashboard. Ignore unless explicitly asked.
