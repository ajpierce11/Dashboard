# Changelog

Notable changes to the Testing Dashboard. Newest at top. Versioning is date-based since there's no semver published.

## 2026-05-13

### Added
- **Metadata filters on AI Assistant** — multiselect for product / model / endpoint gates all retrieval (fulltext, vector, keyword fallback) so questions stay scoped. "Methods we've used for collagen endpoints in rat models" only pulls matching studies.
- **🧠 AI metadata enrichment layer** — new `ai_metadata.py` module walks the library and has the LLM extract structured scientific fields (product, model, endpoints, timepoints) per study, cached in `Library/library_ai_metadata.json`. Runs incrementally — only new/changed studies are re-processed — so subsequent refreshes are seconds. Sits alongside `library_index.json` and `library_vectors.npz` without replacing either.
- **🗺 Coverage matrix** on the Home tab — heatmap of studies by product × model, built from the AI metadata. Makes "where have we looked, and where haven't we?" a glanceable question. Includes a raw-extraction view so admins can spot-check the AI's tagging.
- **🏠 Home tab** — new first tab with recent library additions (last 14 days), data freshness across workbook / index / vectors, and a welcome line. Replaces "open each tab to see what's there" with a glanceable landing page.
- **⭐ Library bookmarks** — star any document to pin it to your Home tab. Stored per-user under `Library/.user_prefs/<username>.json` so bookmarks follow you across machines.
- **AI follow-up suggestions** — after every AI answer, the model proposes 3 concise follow-up questions as buttons. Click one to queue it as the next turn.
- **Admin gating** via `DASHBOARD_ADMINS` env var. Only listed users see Sync, Rebuild, Build-index, Organize, and Save-to-Library buttons; everyone else gets read-only banners with the admin's contact info.
- **Save generated reports to Library** (admin-only) — Report Generator can drop a .docx straight into `Library/Study Reports/` and trigger a Sync so it joins the index immediately.
- **Export chat as Markdown** button next to Clear conversation.
- **"Copy raw text" expander** under each AI Assistant message — uses `st.code`'s built-in clipboard button so users can paste answers into email/reports as markdown.
- **Build-time pre-flight** — admins see the embed scope and rough ETA ("Update will embed 3 new, remove 0. ~10 s.") before clicking.
- **"New" badge** (✨) on library doc cards whose most recent file is within the last 7 days.
- **Search-match highlighting** — library search highlights the matched substring in each card title.
- **Workbook freshness** shown inline in the header caption ("workbook updated 3h ago").
- **Top-right status chip** showing signed-in user, role (admin/viewer), and AI connection status.
- **AbbVie logo** in the app header and as the browser-tab favicon; falls back to a text-only header if the Assets folder is missing.
- **Troubleshooting + runbook** section in the README covering workbook lock, key rotation, corrupted index, and common failure modes.

### Changed
- **AI Assistant system prompt** now includes explicit cross-study synthesis guidance — organise answers by claim not by study, flag disagreements between studies, respect product-class distinctions (HA-only vs biostimulatory vs regenerative are different categories, not interchangeable), and say so plainly when retrieval was thin so the user knows whether to trust the synthesis as comprehensive.
- **Default retrieval breadth** bumped from 25 → 40 chunks for broad questions without a specific study number, giving synthesis questions more evidence to compare against.
- **Removed the Insights tab.** The coverage matrix / gap analysis / suggest-next-experiments features rewarded the wrong primitive (structural cell-counting) rather than mechanistic synthesis. Kept the AI-metadata cache underneath since the filters and future synthesis features build on it; deleted `insights.py`.
- **Dark mode** with AbbVie brand palette. Near-black background (`#0B1220`), Light Blue text (`#EDF0FF`), Medium Blue (`#A6B5E0`) as the accent. Plotly figures use the `plotly_dark` template; chart palette rebuilt around the lighter brand colors so every series stays visible on dark.
- **Typography** uses a system font stack instead of Streamlit's Source Sans, removing the most recognisable "Streamlit app" tell. Hamburger menu and "Made with Streamlit" footer hidden.
- **Library Sync now also updates the AI vector index in one pass** — a single click handles both indexes. Previously admins had to click Sync and then Update index separately, and forgetting the second step meant new docs were browsable but invisible to semantic search.
- **Starter prompts** in the AI Assistant are derived from the live dataset and library index instead of hardcoded — no more broken examples referring to study numbers that don't exist.
- **Chat error messages** no longer leak ILIAD HTTP response bodies; full details still go to the terminal for admin diagnosis.
- **Library mismatch warning** compares files-to-files (not files-to-entries), so multi-file study groups no longer trigger spurious warnings.
- **Stats computations** (`run_pairwise_ttests`, `run_anova`, `run_fisher_overall`) cached with `@st.cache_data` — reruns with identical selection skip the scipy work.
- **Excel export** is no longer built on every rerun — a **Prepare Excel download** button builds it on demand. Main browse flow stays responsive after selection changes.
- **Plotly scroll-zoom disabled** so charts no longer hijack the page scroll wheel.

### Security / correctness (pre-launch review pass)
- **Admin gating fails closed on CML.** If `DASHBOARD_ADMINS` is unset while running on a CML deployment (detected via `CDSW_*` env vars), nobody is admin until it's configured. Previously an unset variable silently made every teammate an admin — exactly the opposite of safe default for a shared deployment.
- **Sync lockfile.** `_sync_library_and_vectors` now drops a short-TTL lockfile in `Library/` so two admins clicking Sync simultaneously don't overlap and stomp each other's writes.
- **Sync no longer silently triggers a full rebuild.** If the vector `.npz` fails to load (corruption, transient I/O), Sync shows a clear error and points to the Full rebuild button instead of falling through to a surprise 10-minute rebuild.
- **`build_index.py` now writes the same `index_stamp` + atomic save** as `VectorStore._save`. Fixes the "consider rebuilding" nag reappearing permanently after every CLI build.
- **PDF-extract cache** switched from `@st.cache_resource` to `@st.cache_data` so a transient parse failure doesn't poison the cache for the rest of the session. Cache size bumped from 64 to 256 entries.
- **Removed broken `file://` citation links** — they only resolved for the machine running the app; teammates' browsers would hit their own disk or a 404. Plain titles shown instead; a proper share-path link scheme is a future addition.
- **n = 10 default warning** promoted from a small caption to a prominent warning banner, since it silently drives every reported p-value and the Significance column in the Excel export.
- **Internal-study starter prompt.** The "Summarise study X" example now prefers Study Reports / Study Protocols / Test Methods entries, so the example doesn't point at an external publication when an alphabetically earlier paper exists.

### Fixed
- Atomic write for `library_vectors.npz` — two concurrent saves or a crash mid-save can no longer leave a partial file that every reader treats as corrupted.
- `needs_rebuild()` compares the library's logical `last_updated` stamp (now embedded inside the .npz) instead of filesystem mtimes, so the "consider rebuilding" nag only appears when content actually changed.
- Non-admin sessions pick up the admin's vector rebuild automatically — `VectorStore.search` stat-checks the `.npz` before each query and re-reads if disk is newer.
- Workbook-locked errors (someone else has the Excel file open) show a friendly message with a Retry button instead of a technical traceback.
- Report Generator hard-returns when the vector index isn't built, eliminating a `NoneType` crash on generate.
- Chat "follow-up" detection no longer matches bare "it"/"its", which previously treated "is it ok?" as a reference to the prior document.
- Dark-mode contrast fixes: the "revisions" and "preview" chips on library cards now use explicit brand hex (they had been using Streamlit's internal CSS vars that resolved to light-theme colors).
- Plotly hover tooltips get an explicit dark bubble styled with brand colors (were bright-white flashes on the near-black chart background).
- Copy-raw-text code block drops the markdown syntax highlighter — defaults to a neutral dark panel.

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
