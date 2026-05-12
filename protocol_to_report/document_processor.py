"""
document_processor.py
Core logic for reading study protocols, converting tense via spaCy + pyinflect
(fully offline), and generating study reports while preserving formatting.

Token-level run mapping ensures only the specific run containing a changed word
is modified — all other runs keep their formatting untouched.
"""

import os
import re
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

import spacy
import pyinflect  # noqa: F401

_nlp = None


def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_md")
    return _nlp


# ---------------------------------------------------------------------------
# Document structure helpers
# ---------------------------------------------------------------------------

def get_paragraph_style_name(paragraph):
    if paragraph.style:
        return paragraph.style.name
    return "Normal"


def is_heading(paragraph):
    style = get_paragraph_style_name(paragraph)
    return style.startswith("Heading") or style.startswith("Title")


def get_heading_level(paragraph):
    style = get_paragraph_style_name(paragraph)
    if style == "Title":
        return 0
    if style.startswith("Heading"):
        try:
            return int(style.split()[-1])
        except (ValueError, IndexError):
            return 1
    return None


# ---------------------------------------------------------------------------
# Token-level run preservation
# ---------------------------------------------------------------------------

def _build_char_to_run_map(runs):
    char_to_run = []
    for run_idx, run in enumerate(runs):
        for _ in run.text:
            char_to_run.append(run_idx)
    return char_to_run


def apply_token_replacements(runs, full_text, replacements):
    """
    Apply {char_start: (old_text, new_text)} replacements directly to the runs
    that own those character positions. Only touched runs are modified.
    Processes right-to-left so offsets stay valid.
    """
    if not replacements or not runs:
        return False

    run_texts = [run.text for run in runs]
    char_to_run = _build_char_to_run_map(runs)

    run_starts = []
    offset = 0
    for run in runs:
        run_starts.append(offset)
        offset += len(run.text)

    for char_start in sorted(replacements.keys(), reverse=True):
        old_word, new_word = replacements[char_start]

        if char_start >= len(char_to_run):
            continue

        run_idx = char_to_run[char_start]
        local_start = char_start - run_starts[run_idx]
        local_end = local_start + len(old_word)

        current_run_text = run_texts[run_idx]

        if local_end <= len(current_run_text) and current_run_text[local_start:local_end] == old_word:
            run_texts[run_idx] = (
                current_run_text[:local_start] + new_word + current_run_text[local_end:]
            )
            length_diff = len(new_word) - len(old_word)
            if length_diff != 0:
                for r in range(run_idx + 1, len(run_starts)):
                    run_starts[r] += length_diff
                char_to_run = []
                for r_idx, r_text in enumerate(run_texts):
                    for _ in r_text:
                        char_to_run.append(r_idx)
        else:
            # Token spans run boundary — targeted find/replace
            joined = "".join(run_texts)
            idx = joined.find(old_word, max(0, char_start - 5))
            if idx >= 0:
                new_joined = joined[:idx] + new_word + joined[idx + len(old_word):]
                _fallback_redistribute(runs, run_texts, new_joined)
                return True

    changed = False
    for run, new_text in zip(runs, run_texts):
        if run.text != new_text:
            run.text = new_text
            changed = True

    return changed


def _fallback_redistribute(runs, run_texts, new_full_text):
    if len(runs) == 1:
        runs[0].text = new_full_text
        return

    old_lengths = [len(t) for t in run_texts]
    total_old = sum(old_lengths)

    if total_old == 0:
        runs[0].text = new_full_text
        for run in runs[1:]:
            run.text = ""
        return

    pos = 0
    for i, run in enumerate(runs):
        if i == len(runs) - 1:
            run.text = new_full_text[pos:]
        else:
            proportion = old_lengths[i] / total_old
            chars = max(0, round(proportion * len(new_full_text)))
            run.text = new_full_text[pos:pos + chars]
            pos += chars


# ---------------------------------------------------------------------------
# Tense conversion — spaCy + pyinflect + regex fallbacks
# ---------------------------------------------------------------------------

_MODALS_TO_REMOVE = {"will", "shall"}

_DEFINITION_SIGNALS = {
    "defined as", "refers to", "known as", "means that", "stands for",
    "abbreviated as", "is a measure of", "is the ratio", "is defined",
    "are defined", "is characterized", "represents a", "denotes",
}

_PLURAL_PRONOUNS = {"they", "we", "these", "those", "both", "all", "some"}


def _check_clause_for_definition(token):
    clause_root = token
    depth = 0
    while (clause_root.dep_ not in ("ROOT", "conj", "advcl", "relcl", "ccomp")
           and clause_root.head != clause_root and depth < 10):
        clause_root = clause_root.head
        depth += 1

    clause_text = " ".join(t.text.lower() for t in clause_root.subtree)
    return any(sig in clause_text for sig in _DEFINITION_SIGNALS)


def _preserve_case(original, replacement):
    if not original or not replacement:
        return replacement
    if original.isupper() and len(original) > 1:
        return replacement.upper()
    if original[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement


def _get_past_tense(token):
    past = token._.inflect("VBD")
    if past:
        return _preserve_case(token.text, past)
    return None


def _subject_is_plural(token, tokens):
    """Walk the dependency tree to determine if the verb's subject is plural."""
    search_targets = []
    if token.dep_ in ("aux", "auxpass"):
        search_targets.append(token.head)
    search_targets.append(token)

    for target in search_targets:
        for child in target.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                if child.tag_ in ("NNS", "NNPS"):
                    return True
                if child.text.lower() in _PLURAL_PRONOUNS:
                    return True
                for subchild in child.children:
                    if subchild.dep_ == "conj":
                        return True
                return False

    head = token.head
    if head and head != token:
        for child in head.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                if child.tag_ in ("NNS", "NNPS"):
                    return True
                if child.text.lower() in _PLURAL_PRONOUNS:
                    return True
                for subchild in child.children:
                    if subchild.dep_ == "conj":
                        return True
                return False

    return False


def _get_past_be(token, tokens):
    if _subject_is_plural(token, tokens):
        return "were"
    return "was"


# ---------------------------------------------------------------------------
# Regex-based fallback patterns for constructs spaCy may miss
# ---------------------------------------------------------------------------

def _apply_regex_fallbacks(text):
    """
    Catch common future/present tense patterns that spaCy may not tag correctly.
    Returns (new_text, was_changed).
    """
    original = text
    changed = False

    # "will be <past_participle>" — catch any remaining
    def _will_be_repl(m):
        nonlocal changed
        changed = True
        # Determine was/were based on preceding context (simplified)
        pre = text[:m.start()].lower()
        be_form = "were" if _pre_context_is_plural(pre) else "was"
        return be_form + " " + m.group(2)

    text = re.sub(
        r'\b[Ww]ill\s+be\s+(\w+ed|measured|injected|evaluated|collected|analyzed|performed|conducted|administered|assessed|determined|observed|recorded|noted|prepared|stored|maintained|monitored|tested|completed|used|applied|examined|identified|selected|assigned|treated|included|excluded|obtained|calculated|compared|considered|processed|reviewed|submitted|reported|documented|approved|required)\b',
        lambda m: ("were" if _pre_context_is_plural(text[:m.start()].lower()) else "was") + " " + m.group(1),
        text,
        flags=re.IGNORECASE
    )

    # "will <verb>" (without "be") — simple future
    def _will_verb_repl(m):
        nonlocal changed
        changed = True
        verb = m.group(1)
        # Simple past tense heuristic for regular verbs
        nlp = get_nlp()
        doc = nlp(verb)
        if doc and len(doc) > 0:
            past = doc[0]._.inflect("VBD")
            if past:
                return _preserve_case(verb, past)
        # Fallback for regular verbs
        if verb.endswith("e"):
            return verb + "d"
        return verb + "ed"

    text = re.sub(
        r'\b[Ww]ill\s+(?:not\s+)?(?!be\b)(\w+)\b',
        lambda m: _will_verb_repl(m) if "not" not in m.group(0).lower()
        else "did not " + m.group(1),
        text
    )

    # "shall be" → "was/were"
    text_new = re.sub(
        r'\b[Ss]hall\s+be\b',
        lambda m: "were" if _pre_context_is_plural(text[:m.start()].lower()) else "was",
        text
    )
    if text_new != text:
        changed = True
        text = text_new

    if text != original:
        changed = True

    return text, changed


def _pre_context_is_plural(pre_text):
    """Simple heuristic: check if the last noun-like word before the verb is plural."""
    words = pre_text.strip().split()
    if not words:
        return False
    # Check last few words for plural indicators
    for word in reversed(words[-5:]):
        word = word.strip(",.;:()")
        if word in _PLURAL_PRONOUNS:
            return True
        if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
            # Likely plural noun (heuristic)
            if word not in ("this", "was", "has", "is", "its", "us", "thus", "plus",
                            "status", "analysis", "basis", "diagnosis", "process",
                            "address", "access", "across", "less", "unless"):
                return True
    return False


# ---------------------------------------------------------------------------
# Main paragraph conversion
# ---------------------------------------------------------------------------

def convert_paragraph_preserving_runs(para):
    """
    Convert a paragraph from present/future to past tense.
    Uses spaCy for primary conversion + regex fallbacks for missed patterns.
    Only modifies specific runs containing changed words.

    Returns (was_changed, original_text, converted_text).
    """
    nlp = get_nlp()
    full_text = para.text

    if not full_text.strip():
        return False, full_text, full_text

    doc = nlp(full_text)
    tokens = list(doc)
    n = len(tokens)

    if n == 0:
        return False, full_text, full_text

    replacements = {}

    i = 0
    while i < n:
        tok = tokens[i]
        tok_lower = tok.text.lower()

        if _check_clause_for_definition(tok):
            i += 1
            continue

        # Pattern 1: "will/shall" + verb phrase
        if tok_lower in _MODALS_TO_REMOVE and tok.pos_ in ("AUX", "VERB", "NOUN"):
            # spaCy sometimes mis-tags "will" — also check by text
            j = i + 1
            # Skip adverbs and "not"
            while j < n and (tokens[j].pos_ == "ADV" or tokens[j].text.lower() == "not"):
                j += 1

            if j < n:
                next_tok = tokens[j]
                next_lower = next_tok.text.lower()

                if next_lower == "be" and j + 1 < n:
                    past_be = _get_past_be(tok, tokens)

                    # Handle "will not be"
                    has_not = any(tokens[k].text.lower() == "not" for k in range(i + 1, j))

                    # Delete modal + trailing space
                    del_end = tok.idx + len(tok.text)
                    trailing = " " if del_end < len(full_text) and full_text[del_end] == " " else ""
                    replacements[tok.idx] = (tok.text + trailing, "")

                    replacements[next_tok.idx] = (next_tok.text, _preserve_case(next_tok.text, past_be))
                    i = j + 1
                    continue

                elif next_lower == "have" and j + 1 < n and tokens[j + 1].tag_ == "VBN":
                    del_end = tok.idx + len(tok.text)
                    trailing = " " if del_end < len(full_text) and full_text[del_end] == " " else ""
                    replacements[tok.idx] = (tok.text + trailing, "")
                    replacements[next_tok.idx] = (next_tok.text, _preserve_case(next_tok.text, "had"))
                    i = j + 1
                    continue

                elif next_tok.pos_ in ("VERB", "AUX") or next_tok.tag_ == "VB":
                    past = _get_past_tense(next_tok)
                    if past:
                        del_end = tok.idx + len(tok.text)
                        trailing = " " if del_end < len(full_text) and full_text[del_end] == " " else ""
                        replacements[tok.idx] = (tok.text + trailing, "")
                        replacements[next_tok.idx] = (next_tok.text, past)
                        i = j + 1
                        continue

            i += 1
            continue

        # Pattern 2: "is/are/am" as auxiliary or copula
        if tok_lower in ("is", "are", "am") and tok.pos_ == "AUX":
            past_be = _get_past_be(tok, tokens)
            replacements[tok.idx] = (tok.text, _preserve_case(tok.text, past_be))
            i += 1
            continue

        # Pattern 3: "has/have" as auxiliary
        if tok_lower in ("has", "have") and tok.pos_ == "AUX":
            j = i + 1
            while j < n and tokens[j].pos_ == "ADV":
                j += 1
            if j < n and tokens[j].tag_ == "VBN":
                replacements[tok.idx] = (tok.text, _preserve_case(tok.text, "had"))
                i = j + 1
                continue

        # Pattern 4: "does/do" + base verb
        if tok_lower in ("does", "do") and tok.pos_ == "AUX":
            j = i + 1
            while j < n and tokens[j].pos_ in ("ADV", "PART"):
                j += 1
            if j < n and tokens[j].pos_ == "VERB" and tokens[j].tag_ == "VB":
                replacements[tok.idx] = (tok.text, _preserve_case(tok.text, "did"))
                i = j + 1
                continue

        # Pattern 5: Present tense main verbs (VBZ, VBP)
        if tok.tag_ in ("VBZ", "VBP") and tok.pos_ == "VERB":
            past = _get_past_tense(tok)
            if past and past.lower() != tok_lower:
                replacements[tok.idx] = (tok.text, past)

        i += 1

    # Apply spaCy-based replacements to runs
    spacy_changed = False
    if replacements:
        spacy_changed = apply_token_replacements(para.runs, full_text, replacements)

    # Clean up double spaces
    for run in para.runs:
        if "  " in run.text:
            run.text = re.sub(r"  +", " ", run.text)

    # Regex fallback pass on the current text for anything spaCy missed
    current_text = "".join(run.text for run in para.runs)
    fallback_text, fallback_changed = _apply_regex_fallbacks(current_text)

    if fallback_changed and fallback_text != current_text:
        # Apply fallback changes — need to update runs
        # Use simple proportional redistribution for regex changes
        if len(para.runs) == 1:
            para.runs[0].text = fallback_text
        else:
            # Try to apply as a targeted replacement
            _apply_text_diff_to_runs(para.runs, current_text, fallback_text)

    final_text = "".join(run.text for run in para.runs)
    total_changed = spacy_changed or (final_text != full_text)

    return total_changed, full_text, final_text


def _apply_text_diff_to_runs(runs, old_text, new_text):
    """Apply a text diff to runs by finding changed regions."""
    from difflib import SequenceMatcher

    if len(runs) == 1:
        runs[0].text = new_text
        return

    sm = SequenceMatcher(None, old_text, new_text, autojunk=False)
    char_to_run = _build_char_to_run_map(runs)

    run_texts = [run.text for run in runs]
    run_starts = []
    offset = 0
    for run in runs:
        run_starts.append(offset)
        offset += len(run.text)

    # Process each change
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue

        if i1 < len(char_to_run):
            run_idx = char_to_run[i1]
            local_start = i1 - run_starts[run_idx]
            local_end = local_start + (i2 - i1)
            current = run_texts[run_idx]
            replacement = new_text[j1:j2]
            run_texts[run_idx] = current[:local_start] + replacement + current[local_end:]

    for run, new_t in zip(runs, run_texts):
        run.text = new_t


# ---------------------------------------------------------------------------
# Document insertion helpers
# ---------------------------------------------------------------------------

def _find_reference_section_index(doc):
    """
    Find the paragraph index of the References/Bibliography heading.
    Returns None if not found.
    """
    for i, para in enumerate(doc.paragraphs):
        if is_heading(para):
            text = para.text.strip().lower()
            if text in ("references", "reference", "bibliography", "works cited",
                         "literature cited", "citations"):
                return i
    return None


def _insert_paragraph_before(doc, ref_para, text, style_name=None):
    """
    Insert a new paragraph immediately before ref_para in the document.
    Returns the new paragraph.
    """
    new_para = doc.add_paragraph()  # Add at end temporarily

    # Move the new paragraph's XML element before the reference paragraph
    ref_para._element.addprevious(new_para._element)

    new_para.text = text
    if style_name:
        try:
            new_para.style = doc.styles[style_name]
        except KeyError:
            pass

    return new_para


def _insert_image_before(doc, ref_para, img_path, width_inches=5.5):
    """Insert an image paragraph before ref_para."""
    new_para = doc.add_paragraph()
    ref_para._element.addprevious(new_para._element)
    new_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = new_para.add_run()
    try:
        run.add_picture(img_path, width=Inches(width_inches))
    except Exception as e:
        run.text = f"[Image: {os.path.basename(img_path)}]"
    return new_para


# ---------------------------------------------------------------------------
# Main document processing class
# ---------------------------------------------------------------------------

class ProtocolProcessor:

    def __init__(self, filepath):
        self.filepath = filepath
        self.doc = Document(filepath)
        self.sections = self._parse_sections()
        self.conversion_log = []

    def _parse_sections(self):
        sections = []
        current_section = {
            "heading": "(Document Start)",
            "heading_level": 0,
            "para_indices": [],
        }

        for i, para in enumerate(self.doc.paragraphs):
            if is_heading(para) and para.text.strip():
                if current_section["para_indices"] or current_section["heading"] != "(Document Start)":
                    sections.append(current_section)
                current_section = {
                    "heading": para.text.strip(),
                    "heading_level": get_heading_level(para) or 1,
                    "para_indices": [i],
                }
            else:
                current_section["para_indices"].append(i)

        if current_section["para_indices"]:
            sections.append(current_section)

        return sections

    def get_structure(self):
        structure = []
        for sec in self.sections:
            body_texts = []
            for idx in sec["para_indices"]:
                para = self.doc.paragraphs[idx]
                if not is_heading(para) and para.text.strip():
                    body_texts.append(para.text.strip()[:150])
            structure.append({
                "heading": sec["heading"],
                "level": sec["heading_level"],
                "paragraph_count": len(body_texts),
                "preview": body_texts[:3],
            })
        return structure

    def replace_protocol_with_report(self):
        """
        Replace 'Protocol' with 'Technical Report' in the document title area
        (first few paragraphs and headings) and in headers/footers.
        Returns the number of replacements made.
        """
        count = 0

        # Check the first 10 paragraphs (title area)
        for para in self.doc.paragraphs[:10]:
            for run in para.runs:
                if "Protocol" in run.text or "PROTOCOL" in run.text:
                    run.text = run.text.replace("Protocol", "Technical Report")
                    run.text = run.text.replace("PROTOCOL", "TECHNICAL REPORT")
                    count += 1

        # Check headers and footers in all sections
        for section in self.doc.sections:
            for header in [section.header, section.first_page_header]:
                if header and header.is_linked_to_previous is False or header:
                    try:
                        for para in header.paragraphs:
                            for run in para.runs:
                                if "Protocol" in run.text or "PROTOCOL" in run.text:
                                    run.text = run.text.replace("Protocol", "Technical Report")
                                    run.text = run.text.replace("PROTOCOL", "TECHNICAL REPORT")
                                    count += 1
                    except Exception:
                        pass

            for footer in [section.footer, section.first_page_footer]:
                if footer:
                    try:
                        for para in footer.paragraphs:
                            for run in para.runs:
                                if "Protocol" in run.text or "PROTOCOL" in run.text:
                                    run.text = run.text.replace("Protocol", "Technical Report")
                                    run.text = run.text.replace("PROTOCOL", "TECHNICAL REPORT")
                                    count += 1
                    except Exception:
                        pass

        return count

    def convert_to_past_tense(self, progress_callback=None):
        paragraphs = list(self.doc.paragraphs)

        table_paragraphs = []
        for table in self.doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    for para in cell.paragraphs:
                        if para.text.strip():
                            table_paragraphs.append(para)

        all_paragraphs = paragraphs + table_paragraphs
        total = len(all_paragraphs)
        changes_made = 0

        _ = get_nlp()

        for i, para in enumerate(all_paragraphs):
            if not para.text.strip():
                if progress_callback and (i + 1) % 10 == 0:
                    progress_callback(i + 1, total, changes_made)
                continue

            if is_heading(para):
                if progress_callback and (i + 1) % 10 == 0:
                    progress_callback(i + 1, total, changes_made)
                continue

            was_changed, original, converted = convert_paragraph_preserving_runs(para)

            if was_changed:
                changes_made += 1
                self.conversion_log.append({
                    "original": original[:100],
                    "converted": converted[:100],
                })

            if progress_callback and (i + 1) % 10 == 0:
                progress_callback(i + 1, total, changes_made)

        if progress_callback:
            progress_callback(total, total, changes_made)

        return {
            "total_paragraphs": total,
            "changes_made": changes_made,
            "log": self.conversion_log,
        }

    def add_report_sections(self, sections_data, heading_level=1):
        """
        Add report sections (Deviations, Results, Conclusions) to the document.
        Inserts BEFORE the References section if one exists, otherwise appends.

        sections_data is a list of dicts:
        [
            {
                "title": "Deviations",
                "content": "...",
                "subsections": [
                    {
                        "title": "Subsection Title",
                        "content": "...",
                        "items": [
                            {"image_path": "/path/to/img.png", "caption": "Figure 1: ..."},
                            ...
                        ]
                    },
                    ...
                ]
            },
            ...
        ]
        """
        ref_idx = _find_reference_section_index(self.doc)

        if ref_idx is not None:
            ref_para = self.doc.paragraphs[ref_idx]
            self._insert_sections_before(ref_para, sections_data, heading_level)
        else:
            self._append_sections(sections_data, heading_level)

    def _get_heading_style(self, level):
        style_name = f"Heading {level}"
        try:
            return self.doc.styles[style_name]
        except KeyError:
            try:
                return self.doc.styles["Heading 1"]
            except KeyError:
                return None

    def _insert_sections_before(self, ref_para, sections_data, heading_level):
        """Insert all sections before the reference paragraph."""
        for section in sections_data:
            if not section.get("title"):
                continue

            # Main heading
            h_para = _insert_paragraph_before(
                self.doc, ref_para, section["title"],
                f"Heading {heading_level}"
            )
            style = self._get_heading_style(heading_level)
            if style:
                h_para.style = style

            # Main content
            if section.get("content"):
                for line in section["content"].split("\n"):
                    stripped = line.strip()
                    if stripped:
                        p = _insert_paragraph_before(self.doc, ref_para, "", "Normal")
                        self._set_paragraph_text_with_list_detection(p, stripped)

            # Subsections
            for sub in section.get("subsections", []):
                if not sub.get("title"):
                    continue

                # Subsection heading (one level deeper)
                sub_level = min(heading_level + 1, 9)
                sh_para = _insert_paragraph_before(
                    self.doc, ref_para, sub["title"],
                    f"Heading {sub_level}"
                )
                sub_style = self._get_heading_style(sub_level)
                if sub_style:
                    sh_para.style = sub_style

                # Subsection content
                if sub.get("content"):
                    for line in sub["content"].split("\n"):
                        stripped = line.strip()
                        if stripped:
                            p = _insert_paragraph_before(self.doc, ref_para, "", "Normal")
                            self._set_paragraph_text_with_list_detection(p, stripped)

                # Items (images + captions)
                for item in sub.get("items", []):
                    img_path = item.get("image_path")
                    caption = item.get("caption", "")

                    if img_path and os.path.exists(img_path):
                        _insert_image_before(self.doc, ref_para, img_path)

                    if caption:
                        cap_para = _insert_paragraph_before(
                            self.doc, ref_para, caption, "Normal"
                        )
                        cap_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                        # Make caption slightly smaller and italic
                        for run in cap_para.runs:
                            run.italic = True
                            run.font.size = Pt(10)

    def _append_sections(self, sections_data, heading_level):
        """Append sections to the end of the document (fallback)."""
        for section in sections_data:
            if not section.get("title"):
                continue

            h_para = self.doc.add_paragraph(section["title"])
            style = self._get_heading_style(heading_level)
            if style:
                h_para.style = style

            if section.get("content"):
                for line in section["content"].split("\n"):
                    stripped = line.strip()
                    if stripped:
                        p = self.doc.add_paragraph()
                        self._set_paragraph_text_with_list_detection(p, stripped)

            for sub in section.get("subsections", []):
                if not sub.get("title"):
                    continue

                sub_level = min(heading_level + 1, 9)
                sh_para = self.doc.add_paragraph(sub["title"])
                sub_style = self._get_heading_style(sub_level)
                if sub_style:
                    sh_para.style = sub_style

                if sub.get("content"):
                    for line in sub["content"].split("\n"):
                        stripped = line.strip()
                        if stripped:
                            p = self.doc.add_paragraph()
                            self._set_paragraph_text_with_list_detection(p, stripped)

                for item in sub.get("items", []):
                    img_path = item.get("image_path")
                    caption = item.get("caption", "")

                    if img_path and os.path.exists(img_path):
                        p = self.doc.add_paragraph()
                        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                        run = p.add_run()
                        try:
                            run.add_picture(img_path, width=Inches(5.5))
                        except Exception:
                            run.text = f"[Image: {os.path.basename(img_path)}]"

                    if caption:
                        cap = self.doc.add_paragraph(caption)
                        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                        for run in cap.runs:
                            run.italic = True
                            run.font.size = Pt(10)

    def _set_paragraph_text_with_list_detection(self, para, text):
        """Set paragraph text, detecting bullet/numbered list formatting."""
        if re.match(r"^[-\u2022*]\s+", text):
            clean = re.sub(r"^[-\u2022*]\s+", "", text)
            para.text = clean
            try:
                para.style = self.doc.styles["List Bullet"]
            except KeyError:
                para.text = text
                try:
                    para.style = self.doc.styles["Normal"]
                except KeyError:
                    pass
        elif re.match(r"^\d+[.)]\s+", text):
            clean = re.sub(r"^\d+[.)]\s+", "", text)
            para.text = clean
            try:
                para.style = self.doc.styles["List Number"]
            except KeyError:
                para.text = text
                try:
                    para.style = self.doc.styles["Normal"]
                except KeyError:
                    pass
        else:
            para.text = text
            try:
                para.style = self.doc.styles["Normal"]
            except KeyError:
                pass

    def save_report(self, output_path):
        self.doc.save(output_path)
        return output_path

    def get_paragraph_count(self):
        count = len([p for p in self.doc.paragraphs if p.text.strip()])
        for table in self.doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    count += len([p for p in cell.paragraphs if p.text.strip()])
        return count
