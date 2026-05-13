# Changelog

Notable changes to the Testing Dashboard. Newest at top. Versioning is date-based since there's no semver published.

## 2026-05-13 (latest)

### Changed
- **Removed the left sidebar.** It was eating horizontal space for info most users didn't need at a glance. The two pieces worth always-visible — signed-in user + AI connection status — moved to a compact top-right chip in the header row.
- **Library Sync now also updates the AI vector index in one pass.** Previously admins had to click Sync in the Library tab *and* Update index in the AI Assistant tab separately, and forgetting the second step meant new docs were browsable but invisible to semantic search. A single Sync click now does both.

### Added
- **Sidebar** with at-a-glance status — signed-in user + role (admin/viewer), data freshness (workbook, library index, vector index last-updated), and AI connection status.
- **"New" badge** (✨) on library doc cards whose most recent file is within the last 7 days. Makes fresh material visually obvious.
- **Search-match highlighting** — library search now visually highlights the matched substring in each card title.
- **"Copy raw text" expander** under each AI Assistant message — uses `st.code`'s built-in clipboard button so users can paste answers into email/reports as markdown.
- **Workbook freshness** shown inline in the header caption ("workbook updated 3h ago").

### Changed
- Stats computations (`run_pairwise_ttests`, `run_anova`, `run_fisher_overall`) now cached with `@st.cache_data` — reruns with identical selection skip the scipy work.
- Excel export is no longer built on every rerun — there's a **Prepare Excel download** button that builds on demand, then the download button appears. Main browse flow stays responsive after selection changes.

## 2026-05-13 (even later)

### Changed
- **Switched to dark mode.** Default theme is now dark with AbbVie brand palette inverted: near-black background (`#0B1220`, a deepened Dark Blue), Light Blue text (`#EDF0FF`), Medium Blue (`#A6B5E0`) as the accent / primary-button color. Secondary background is a slate (`#1A2438`).
- **Logo swap** — header uses the white AbbVie wordmark (`AbbVieLogo_white.png`), favicon uses the dark-blue-background variant, so the mark is visible against dark chrome.
- **Chart palette** dropped Dark Blue (disappeared into the background) and swapped in the lighter secondary colors — Medium Blue, Light Red, Light Cobalt, Light Green, Light Purple, Light Copper. Plotly figures now use the `plotly_dark` template so axis labels and legends are light.

## 2026-05-13 (later)

### Added
- **AbbVie logo** in the app header (left of the title) and as the browser-tab favicon. Loaded from `Assets/`; if the file is missing the app falls back to the text-only header automatically so local dev still works.

### Changed
- **AbbVie brand theming pass.** Primary color is now AbbVie Dark Blue (`#071D49`), secondary background is Light Blue (`#EDF0FF`), body text is Dark Gray (`#4B4C4E`) per the brand guide. Titles use the brand Dark Blue with a Medium Blue accent bar.
- **Chart palette** switched from Plotly D3 to AbbVie brand colors — Dark Blue, Dark Cobalt, Dark Red, Dark Green, Dark Purple, Dark Copper (and two lighter blues as fallback). Cleaner and consistent with company visual identity.
- **Typography** uses a system font stack instead of Streamlit's default Source Sans, removing the most recognisable "Streamlit app" tell.
- **Chrome cleanup** — hid the hamburger menu and "Made with Streamlit" footer so the app reads as internal tooling.
- Tab bar, buttons, and expanders restyled to match the brand palette.

## 2026-05-13

### Added
- **Admin gating** via `DASHBOARD_ADMINS` env var. Only listed users see Sync, Rebuild, Build-index, Organize, and Save-to-Library buttons; everyone else gets read-only banners with the admin's contact info.
- **Save generated reports to Library** (admin-only) — Report Generator can now drop a .docx straight into `Library/Study Reports/` and trigger an incremental Sync so it joins the index immediately.
- **Export chat as Markdown** button next to Clear conversation.
- **Clickable source links** — each cited document in an AI answer is now a `file://` link that opens in Word/Acrobat from the shared drive.
- **Build-time pre-flight** — admins see the embed scope and rough ETA ("Update will embed 3 new, remove 0. ~10 s.") before clicking.
- **Troubleshooting + runbook** section in the README covering workbook lock, key rotation, corrupted index, and common failure modes.

### Changed
- Starter prompts in the AI Assistant are now derived from the live dataset and library index instead of hardcoded — no more broken examples referring to study numbers that don't exist.
- Chat error messages no longer leak ILIAD HTTP response bodies; full details still go to the terminal for admin diagnosis.
- Library mismatch warning compares files-to-files (not files-to-entries), so multi-file study groups no longer trigger spurious warnings.

### Fixed
- Atomic write for `library_vectors.npz` — two concurrent saves or a crash mid-save can no longer leave a partial file that every subsequent load treats as corrupted.
- `needs_rebuild()` compares the library's logical `last_updated` stamp (now embedded inside the .npz) instead of filesystem mtimes, so the "consider rebuilding" nag only appears when the index content actually changed.
- Non-admin sessions pick up the admin's vector rebuild automatically — `VectorStore.search` now stat-checks the .npz before each query and re-reads if disk is newer.
- Workbook-locked errors (someone else has the Excel file open) now show a friendly message with a Retry button instead of a technical traceback.
- Report Generator hard-returns when the vector index isn't built, eliminating a `NoneType` crash on generate.
- Chat "follow-up" detection no longer matches bare "it"/"its", which previously treated "is it ok?" as a reference to the prior document.

## 2026-05-12

### Added
- Shared `iliad_client.py` module consolidating chat + embedding calls, retry logic, and corporate-cert handling.
- Shared `title_utils.py` with `base_title`, `strip_version_suffix`, `clean_stem`, and `display_title_from_filename` — previously these were four near-identical copies scattered across files.
- Keyword retrieval (`_retrieve_library_context`) now returns chunk dicts matching the vector-search shape, so source lists show individual documents rather than a single "Keyword search results" tile.

### Changed
- `load_data` cache keyed on workbook mtime — edits on the network drive invalidate the cache automatically.
- `group_documents` blocks by first filename token before running SequenceMatcher similarity, turning the n² scan into per-block scans.
- `_cached_extract_text` moved off `st.session_state` onto a bounded `st.cache_resource` (max 64 entries) to prevent unbounded pickling.
- Path-traversal guard in `_apply_organize_plan`.

### Fixed
- `VectorStore.search` guard clause that was logically inverted — fresh instances now load vectors from disk correctly.
- `search()`'s O(N²) chunk-score scan replaced with O(1) dict lookup; noticeable speedup on queries.
- Dead `_bg` background-rebuild closure removed (never started; would have corrupted Streamlit session state if enabled).
- `build_index.py` no longer hardcodes a per-user Windows path.
- `build_index.py` no longer wraps each embed batch in a leaky daemon thread — relies on `iliad_client` HTTP timeouts and a per-doc elapsed budget.
- `_chunk_text` now warns when a document exceeds the 80-chunk cap instead of silently truncating.

## 2026-05-11

### Added
- Initial GitHub publish.
- Env-var-driven `config.py` so teammates don't have to edit source for their paths.
- `.env.example`, `.gitignore`, `requirements.txt`, and README — standard repo scaffolding.

### Removed
- `protocol_to_report/` (separate Flask app, not part of the dashboard).
- `check_index.py`, `check_vectors.py` diagnostic scripts.
- `library_index.py` (unused; comparator.py reimplemented the same logic inline).
