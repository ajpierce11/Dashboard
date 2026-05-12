"""
Product Performance Comparator
AbbVie – Soft-tissue/material testing dashboard
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import ttest_ind_from_stats, f_oneway, combine_pvalues
import numpy as np
from io import BytesIO
import requests
from vector_store import VectorStore
import iliad_client
import os
import re
import json
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

from config import FILE_PATH, LIBRARY_PATH
from title_utils import base_title, clean_stem as _shared_clean_stem, display_title_from_filename

STATIC_PROPERTIES = [
    "HA Concentration (mg/mL)",
    "Elasticity (G', Pa)",
    "Cohesivity (gmf)",
    "Water Uptake (%)",
    "Extrusion Force (50mm/min(N))",
]

# Plotly color sequence – professional, print-friendly
COLOR_SEQUENCE = px.colors.qualitative.D3


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _workbook_mtime() -> float:
    try:
        return Path(FILE_PATH).stat().st_mtime
    except OSError:
        return 0.0


@st.cache_data
def _load_data_cached(mtime: float) -> tuple[pd.DataFrame, list[str], dict[str, str], pd.DataFrame]:
    """Cached by workbook mtime — edits on the network drive invalidate automatically."""
    df_raw = pd.read_excel(FILE_PATH, header=0)

    # --- Identify timepoint and std columns by position ---
    timepoints: list[str] = df_raw.columns[1:7].tolist()
    std_cols: list[str] = df_raw.columns[7:13].tolist()

    # --- Melt to long form ---
    df_avg = df_raw.melt(id_vars=["Product"], value_vars=timepoints,
                         var_name="Timepoint", value_name="Average")
    df_std = df_raw.melt(id_vars=["Product"], value_vars=std_cols,
                         var_name="_tp_dummy", value_name="StdDev")

    df_long = df_avg.copy()
    df_long["StdDev"] = df_std["StdDev"].values
    df_long["Timepoint"] = pd.Categorical(df_long["Timepoint"],
                                           categories=timepoints, ordered=True)
    df_long["Average"] = pd.to_numeric(df_long["Average"], errors="coerce")
    df_long["StdDev"] = pd.to_numeric(df_long["StdDev"], errors="coerce")

    # --- Reference column (col index 14) ---
    df_raw = df_raw.rename(columns={df_raw.columns[14]: "Reference"})
    product_to_ref: dict[str, str] = dict(
        zip(df_raw["Product"], df_raw["Reference"].fillna(""))
    )

    # --- Static properties (cols 15–19) ---
    rename_map = {df_raw.columns[15 + i]: col for i, col in enumerate(STATIC_PROPERTIES)}
    df_raw = df_raw.rename(columns=rename_map)
    product_properties = df_raw[["Product"] + STATIC_PROPERTIES].set_index("Product")

    return df_long, timepoints, product_to_ref, product_properties


def load_data() -> tuple[pd.DataFrame, list[str], dict[str, str], pd.DataFrame]:
    """
    Load and reshape the Excel workbook. Cache is keyed on the workbook's
    mtime so edits on the network drive are picked up without a manual reload.

    Expected column layout (0-indexed):
      0     : Product name
      1–6   : Mean values at each timepoint
      7–12  : Corresponding std deviations
      13    : (unused / spacer)
      14    : Reference / lot number
      15–19 : Static material properties (STATIC_PROPERTIES)
    """
    return _load_data_cached(_workbook_mtime())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_label(product: str, ref: str) -> str:
    return f"{product} ({ref})" if ref else product


def build_pivot_tables(
    filtered: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (mean pivot, std pivot) indexed by Product with Timepoint columns."""
    pivot_avg = filtered.pivot_table(
        index="Product", columns="Timepoint", values="Average", aggfunc="mean", observed=True
    )
    pivot_std = filtered.pivot_table(
        index="Product", columns="Timepoint", values="StdDev", aggfunc="mean", observed=True
    )
    return pivot_avg, pivot_std


def valid_tp_list(
    timepoints: list[str],
    pivot_avg: pd.DataFrame,
    pivot_std: pd.DataFrame,
    selected_products: list[str],
) -> tuple[list[str], list[str]]:
    """
    Return (valid timepoints, skipped timepoints).
    Only looks up products that actually exist in the pivot index — products
    with no time-series data at all (e.g. Ellanse M) are skipped gracefully
    rather than raising a KeyError.
    """
    present = [p for p in selected_products
               if p in pivot_avg.index and p in pivot_std.index]

    valid, skipped = [], []
    for tp in timepoints:
        if tp not in pivot_avg.columns or tp not in pivot_std.columns:
            skipped.append(tp)
            continue
        if not present:
            skipped.append(tp)
            continue
        rows_ok = (
            pivot_avg.loc[present, tp].notna().all()
            and pivot_std.loc[present, tp].notna().all()
        )
        (valid if rows_ok else skipped).append(tp)
    return valid, skipped


def format_mean_std(mean: float, std: float, decimals: int = 3) -> str:
    fmt = f"{{:.{decimals}f}}"
    return f"{fmt.format(mean)} ± {fmt.format(std)}"


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def build_figure(
    filtered: pd.DataFrame,
    pivot_avg: pd.DataFrame,
    pivot_std: pd.DataFrame,
    selected_products: list[str],
    product_to_ref: dict[str, str],
) -> go.Figure:
    fig = go.Figure()
    for i, product in enumerate(selected_products):
        if product not in pivot_avg.index or product not in pivot_std.index:
            continue
        label = make_label(product, product_to_ref.get(product, ""))
        means = pivot_avg.loc[product]
        stds = pivot_std.loc[product]
        color = COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)]

        fig.add_trace(go.Scatter(
            x=means.index.tolist(),
            y=means.values,
            error_y=dict(type="data", array=stds.values, visible=True),
            mode="lines+markers",
            name=label,
            marker=dict(size=7, color=color),
            line=dict(color=color, width=2),
            hovertemplate=(
                f"<b>{label}</b><br>"
                "Timepoint: %{x}<br>"
                "Mean: %{y:.3f}<br>"
                "SD: %{customdata:.3f}<extra></extra>"
            ),
            customdata=stds.values,
        ))

    fig.update_layout(
        title=dict(text="Lift capacity over time", font=dict(size=16)),
        xaxis_title="Timepoint",
        yaxis_title="Average value",
        legend_title="Product (reference)",
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=50, b=40, l=50, r=20),
        yaxis=dict(
            rangemode="tozero",
            gridcolor="rgba(128,128,128,0.15)",
        ),
        xaxis=dict(gridcolor="rgba(128,128,128,0.1)"),
    )
    return fig


# ---------------------------------------------------------------------------
# Properties radar chart
# ---------------------------------------------------------------------------

# Short axis labels so they don't overlap on the radar
PROPERTY_SHORT_LABELS = {
    "HA Concentration (mg/mL)":        "HA conc.",
    "Elasticity (G', Pa)":             "Elasticity",
    "Cohesivity (gmf)":                "Cohesivity",
    "Water Uptake (%)":                "Water uptake",
    "Extrusion Force (50mm/min(N))":   "Extrusion",
}


def _normalise_props(
    product_properties: pd.DataFrame,
    selected_products: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (raw_props, normalised_props) for selected products.

    Normalisation uses the range across ALL products in the dataset (not just
    the selected ones) so that adding/removing a product never rescales the
    polygons of others. Each attribute is scaled independently to [10, 100]
    — the floor of 10 (not 0) ensures every product has a visible polygon
    even when it scores lowest on an attribute.
    """
    # Full-dataset range for stable normalisation
    all_props = product_properties.copy()
    for col in all_props.columns:
        all_props[col] = pd.to_numeric(all_props[col], errors="coerce")

    col_min = all_props.min()
    col_max = all_props.max()
    col_range = (col_max - col_min).where((col_max - col_min) != 0, other=1)

    raw = all_props.loc[[p for p in selected_products if p in all_props.index]].copy()
    # Scale to [10, 100] so lowest-scoring product still shows on the chart
    norm = (((raw - col_min) / col_range) * 90 + 10).round(1)

    return raw, norm


def build_properties_radar(
    product_properties: pd.DataFrame,
    selected_products: list[str],
    product_to_ref: dict[str, str],
) -> go.Figure:
    """
    Radar chart with per-attribute normalisation against the full dataset range.
    Hover always shows true raw values.
    """
    valid = [p for p in selected_products if p in product_properties.index]
    if not valid:
        return None

    raw, norm = _normalise_props(product_properties, valid)

    short_labels = [PROPERTY_SHORT_LABELS.get(a, a) for a in STATIC_PROPERTIES]
    # Close the polygon
    theta = short_labels + [short_labels[0]]

    fig = go.Figure()
    for i, product in enumerate(valid):
        raw_vals  = [raw.loc[product, a]  for a in STATIC_PROPERTIES]
        norm_vals = [norm.loc[product, a] for a in STATIC_PROPERTIES]
        label     = make_label(product, product_to_ref.get(product, ""))
        color     = COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)]

        hover_text = [
            f"<b>{label}</b><br>{PROPERTY_SHORT_LABELS.get(a,a)}: {v:.3g}"
            for a, v in zip(STATIC_PROPERTIES, raw_vals)
        ] + [f"<b>{label}</b><br>{short_labels[0]}: {raw_vals[0]:.3g}"]

        # No fill at all — outlines only. Fills always stack and bury smaller
        # polygons regardless of opacity. Lines render on top of each other
        # cleanly so every product is visible regardless of selection order.
        # Dash patterns provide a second visual cue beyond color alone.
        dash_styles = ["solid", "dash", "dot", "dashdot", "longdash"]
        dash = dash_styles[i % len(dash_styles)]

        fig.add_trace(go.Scatterpolar(
            r=norm_vals + [norm_vals[0]],
            theta=theta,
            fill="none",
            line=dict(color=color, width=2.5, dash=dash),
            marker=dict(size=6, color=color),
            name=label,
            text=hover_text,
            hoverinfo="text",
        ))

    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                range=[0, 100],
                showticklabels=False,
                gridcolor="rgba(128,128,128,0.18)",
                linecolor="rgba(128,128,128,0.18)",
            ),
            angularaxis=dict(
                gridcolor="rgba(128,128,128,0.12)",
                linecolor="rgba(128,128,128,0.18)",
                tickfont=dict(size=12),
                direction="clockwise",
            ),
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.22,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        # Extra margin so axis labels are never clipped
        margin=dict(t=40, b=80, l=80, r=80),
        height=400,
    )
    return fig


def build_properties_bars(
    product_properties: pd.DataFrame,
    selected_products: list[str],
    product_to_ref: dict[str, str],
) -> go.Figure:
    """
    Small-multiples horizontal bar chart: one subplot per attribute, each with
    its own independent axis. This avoids the problem of a shared Y axis where
    G' (hundreds of Pa) visually swamps HA concentration (tens of mg/mL).
    """
    from plotly.subplots import make_subplots

    valid = [p for p in selected_products if p in product_properties.index]
    if not valid:
        return None

    raw, _ = _normalise_props(product_properties, valid)
    n_attrs = len(STATIC_PROPERTIES)

    fig = make_subplots(
        rows=1,
        cols=n_attrs,
        shared_yaxes=False,
        horizontal_spacing=0.06,
    )

    labels = [make_label(p, product_to_ref.get(p, "")) for p in valid]
    colors = [COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)] for i in range(len(valid))]

    for col_idx, attr in enumerate(STATIC_PROPERTIES, start=1):
        short = PROPERTY_SHORT_LABELS.get(attr, attr)
        vals  = [raw.loc[p, attr] for p in valid]

        for row_idx, (product, val, label, color) in enumerate(
            zip(valid, vals, labels, colors)
        ):
            fig.add_trace(
                go.Bar(
                    x=[label],
                    y=[val],
                    marker_color=color,
                    showlegend=False,
                    text=[f"{val:.3g}"],
                    textposition="outside",
                    hovertemplate=f"<b>{label}</b><br>{short}: {val:.3g}<extra></extra>",
                    cliponaxis=False,
                ),
                row=1,
                col=col_idx,
            )

        # Attribute name as subplot title substitute (xaxis title)
        fig.update_xaxes(
            title_text=short,
            title_font=dict(size=10),
            showticklabels=False,
            row=1,
            col=col_idx,
        )
        fig.update_yaxes(
            gridcolor="rgba(128,128,128,0.15)",
            showticklabels=True,
            tickfont=dict(size=9),
            row=1,
            col=col_idx,
        )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=10, l=40, r=10),
        height=220,
        bargap=0.25,
    )
    return fig


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def build_excel_export(
    filtered: pd.DataFrame,
    pivot_avg: pd.DataFrame,
    pivot_std: pd.DataFrame,
    product_properties: pd.DataFrame,
    selected_products: list[str],
    timepoints: list[str],
    product_to_ref: dict[str, str],
) -> BytesIO:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        wb = writer.book

        # --- Sheet 1: raw data for chart ---
        wide = pivot_avg.reset_index()
        wide["Product"] = wide["Product"].apply(
            lambda p: make_label(p, product_to_ref.get(p, ""))
        )
        wide.to_excel(writer, sheet_name="Chart data", index=False)

        ws = writer.sheets["Chart data"]
        chart = wb.add_chart({"type": "line"})
        n_tp = len(timepoints)
        for row_i, product_label in enumerate(wide["Product"]):
            chart.add_series({
                "name":       ["Chart data", row_i + 1, 0],
                "categories": ["Chart data", 0, 1, 0, n_tp],
                "values":     ["Chart data", row_i + 1, 1, row_i + 1, n_tp],
                "marker":     {"type": "circle", "size": 5},
            })
        chart.set_title({"name": "Lift capacity"})
        chart.set_x_axis({"name": "Timepoint"})
        chart.set_y_axis({"name": "Average"})
        chart.set_style(2)
        ws.insert_chart("H2", chart, {"x_offset": 20, "y_offset": 10})

        # --- Sheet 2: mean ± SD summary (formatted, human-readable) ---
        summary = pd.DataFrame(index=pivot_avg.index)
        for col in pivot_avg.columns:
            summary[col] = [
                format_mean_std(pivot_avg.loc[p, col], pivot_std.loc[p, col])
                if pd.notna(pivot_avg.loc[p, col]) else "N/A"
                for p in pivot_avg.index
            ]
        summary.to_excel(writer, sheet_name="Summary (mean ± SD)")

        # --- Sheet 3: raw numeric mean and SD for further analysis ---
        raw_mean = pivot_avg.copy().round(4)
        raw_sd   = pivot_std.copy().round(4)
        raw_mean.to_excel(writer, sheet_name="Raw mean values")
        raw_sd.to_excel(writer, sheet_name="Raw SD values")

        # --- Sheet 4: product properties ---
        props = product_properties.loc[
            [p for p in selected_products if p in product_properties.index]
        ].copy()
        for col in props.columns:
            props[col] = pd.to_numeric(props[col], errors="coerce").round(3)
        props.to_excel(writer, sheet_name="Product properties")

    output.seek(0)
    return output


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def run_pairwise_ttests(
    pivot_avg: pd.DataFrame,
    pivot_std: pd.DataFrame,
    selected_products: list[str],
    valid_timepoints: list[str],
    n_per_group: int,
) -> dict[str, pd.DataFrame]:
    """Return per-timepoint DataFrames of pairwise t-test results."""
    results = {}
    present = [p for p in selected_products if p in pivot_avg.index and p in pivot_std.index]
    pairs = [
        (present[i], present[j])
        for i in range(len(present))
        for j in range(i + 1, len(present))
    ]
    for tp in valid_timepoints:
        rows = []
        for p1, p2 in pairs:
            try:
                _, p_val = ttest_ind_from_stats(
                    pivot_avg.loc[p1, tp], pivot_std.loc[p1, tp], n_per_group,
                    pivot_avg.loc[p2, tp], pivot_std.loc[p2, tp], n_per_group,
                )
            except Exception:
                p_val = np.nan
            rows.append({
                "Pair": f"{p1} vs {p2}",
                "p-value": f"{p_val:.4f}" if pd.notna(p_val) else "N/A",
                "Significant?": "Yes" if pd.notna(p_val) and p_val < 0.05 else "No",
            })
        results[tp] = pd.DataFrame(rows)
    return results


def run_anova(
    filtered: pd.DataFrame,
    selected_products: list[str],
    valid_timepoints: list[str],
) -> pd.DataFrame:
    """One-way ANOVA across all selected products at each timepoint."""
    rows = []
    for tp in valid_timepoints:
        groups = [
            filtered.loc[filtered["Product"] == p, "Average"].dropna().values
            for p in selected_products
        ]
        try:
            _, p_val = f_oneway(*groups)
        except Exception:
            p_val = np.nan
        rows.append({
            "Timepoint": tp,
            "ANOVA p-value": f"{p_val:.4f}" if pd.notna(p_val) else "N/A",
            "Significant (p < 0.05)?": "Yes" if pd.notna(p_val) and p_val < 0.05 else "No",
        })
    return pd.DataFrame(rows)


def run_fisher_overall(
    pivot_avg: pd.DataFrame,
    pivot_std: pd.DataFrame,
    selected_products: list[str],
    valid_timepoints: list[str],
    n_per_group: int,
) -> pd.DataFrame:
    """
    Summarise overall significance per pair using Fisher's combined probability test.

    Fisher's method combines k independent p-values into a single chi-squared
    statistic: X² = -2 * sum(ln(p_i)), distributed as chi-squared with 2k df.
    This is the statistically correct alternative to averaging p-values —
    it is sensitive to consistent trends across timepoints while properly
    accounting for the number of tests combined.
    """
    present = [p for p in selected_products if p in pivot_avg.index and p in pivot_std.index]
    pairs = [
        (present[i], present[j])
        for i in range(len(present))
        for j in range(i + 1, len(present))
    ]
    rows = []
    for p1, p2 in pairs:
        p_vals = []
        for tp in valid_timepoints:
            try:
                _, p_val = ttest_ind_from_stats(
                    pivot_avg.loc[p1, tp], pivot_std.loc[p1, tp], n_per_group,
                    pivot_avg.loc[p2, tp], pivot_std.loc[p2, tp], n_per_group,
                )
                if pd.notna(p_val) and p_val > 0:
                    p_vals.append(p_val)
            except Exception:
                pass

        if len(p_vals) >= 2:
            _, combined_p = combine_pvalues(p_vals, method="fisher")
        elif len(p_vals) == 1:
            combined_p = p_vals[0]
        else:
            combined_p = np.nan

        rows.append({
            "Pair": f"{p1} vs {p2}",
            "Timepoints combined": len(p_vals),
            "Fisher combined p": f"{combined_p:.4f}" if pd.notna(combined_p) else "N/A",
            "Significant overall?": (
                "Yes" if pd.notna(combined_p) and combined_p < 0.05 else "No"
            ),
        })
    return pd.DataFrame(rows)



# ---------------------------------------------------------------------------
# Document library
# ---------------------------------------------------------------------------

# Mapping from category key → display label
LIBRARY_CATEGORIES = {
    "Test Methods":      "Test Methods",
    "Work Instructions": "Work Instructions",
    "Study Reports":     "Study Reports",
    "Study Protocols":   "Study Protocols",
    "Publications":      "Publications",
}

# Icons shown next to each category tab
CATEGORY_ICONS = {
    "Test Methods":      "🔬",
    "Work Instructions": "📋",
    "Study Reports":     "📊",
    "Study Protocols":   "📝",
    "Publications":      "📖",
}


def classify_document(filename: str, filepath: str | None = None) -> str:
    """
    Classify a document into one of the four library categories.

    Primary rule: if the file lives inside a subfolder named after a category
    (case-insensitive), the subfolder wins. This lets users override auto-
    classification by moving a file — no admin UI required.

    Fallback rules (case-insensitive keyword match on the filename stem):
      - Contains "Protocol"              → Study Protocols
      - Contains "Report" or standalone "TR" → Study Reports
      - Contains standalone "ME" or "TM" → Test Methods
      - Contains standalone "WI"         → Work Instructions
      - Anything else                    → Study Protocols (safest default)
    """
    if filepath:
        try:
            rel = Path(filepath).resolve().relative_to(Path(LIBRARY_PATH).resolve())
            for part in rel.parts[:-1]:  # exclude the filename itself
                for cat in LIBRARY_CATEGORIES:
                    if part.lower() == cat.lower():
                        return cat
        except (ValueError, OSError):
            pass  # filepath outside LIBRARY_PATH — fall through to filename rules

    stem = Path(filename).stem

    if re.search(r"protocol", stem, re.IGNORECASE):
        return "Study Protocols"

    # Study Reports — "Report" anywhere, or "TR" as a standalone token
    # TR (Technical Report) must be checked before ME/TM to avoid misclassification
    if re.search(r"report", stem, re.IGNORECASE):
        return "Study Reports"
    if re.search(r"(?<![A-Za-z])TR(?![A-Za-z])", stem, re.IGNORECASE):
        return "Study Reports"

    # Test methods — ME or TM as standalone tokens
    if re.search(r"(?<![A-Za-z])(ME|TM)(?![A-Za-z])", stem, re.IGNORECASE):
        return "Test Methods"

    # Work instructions — WI as standalone token
    if re.search(r"(?<![A-Za-z])WI(?![A-Za-z])", stem, re.IGNORECASE):
        return "Work Instructions"

    # Default: no recognised keyword → most likely a protocol
    return "Study Protocols"


# Supported file extensions and their display labels / MIME types
SUPPORTED_EXTS: dict[str, dict] = {
    ".pdf":  {"label": "PDF",   "mime": "application/pdf"},
    ".docx": {"label": "Word",  "mime": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"},
    ".xlsx": {"label": "Excel", "mime": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"},
    ".xls":  {"label": "Excel", "mime": "application/vnd.ms-excel"},
    ".pptx": {"label": "PPT",   "mime": "application/vnd.openxmlformats-officedocument.presentationml.presentation"},
    ".ppt":  {"label": "PPT",   "mime": "application/vnd.ms-powerpoint"},
}


# Full text limit stored in index — enough for a 30-page document
# The vector store chunks this into searchable pieces
# No character limit — extract complete document text


# Populated by render_library while a scan is running so that _scan_library_raw
# can report per-file progress back to the UI without taking a callback arg on
# every call site. Set to None when no scan is active.
_scan_progress_callback = None


def _with_timeout(fn, args: tuple, timeout_sec: float):
    """
    Run fn(*args) on a daemon thread and raise TimeoutError if it exceeds
    timeout_sec. Used to keep a single corrupt/huge document from stalling
    the whole library scan forever.
    """
    import threading
    result: list = [None]
    error: list = [None]

    def _target():
        try:
            result[0] = fn(*args)
        except Exception as e:
            error[0] = e

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=timeout_sec)
    if t.is_alive():
        raise TimeoutError(f"operation exceeded {timeout_sec}s")
    if error[0] is not None:
        raise error[0]
    return result[0]


def _extract_full_text(filepath: Path) -> str:
    """
    Extract the complete text from a document file.
    Delegates to doc_text.extract_text (pymupdf-first for PDFs — 3-5× faster
    than pypdf on large documents). Preserves heading markers for docx so
    chunking can split at meaningful boundaries.
    """
    from doc_text import extract_text
    return extract_text(str(filepath))


def _extract_preview(full: str) -> str:
    """
    Return a short content-representative preview, skipping the boilerplate
    (titles, signature lines, all-caps disclaimers) that dominates the first
    page of most study reports. Prefers the first substantive paragraph that
    comes after the first section heading.
    """
    lines = [ln.strip() for ln in full.split("\n") if ln.strip()]

    def _looks_substantive(line: str) -> bool:
        if len(line) < 80:
            return False
        alpha = [c for c in line if c.isalpha()]
        if not alpha:
            return False
        # Skip paragraphs that are mostly uppercase — likely headings or notices
        if sum(1 for c in alpha if c.isupper()) / len(alpha) > 0.5:
            return False
        return True

    after_heading = False
    for line in lines:
        if line.startswith("##"):
            after_heading = True
            continue
        if after_heading and _looks_substantive(line):
            return line[:500]

    # Fallback for docs without headings — first substantive line anywhere
    for line in lines:
        if line.startswith("##"):
            continue
        if _looks_substantive(line):
            return line[:500]

    return ""


def extract_doc_metadata(filepath: Path) -> dict:
    """
    Extract display metadata and full text from a single file.
    Supports .pdf, .docx, .xlsx, .xls, .pptx, .ppt.

    Stores two text fields:
      - preview: first ~500 chars for library card display
      - full_text: complete document text for AI vector search
    """
    stat = filepath.stat()
    ext  = filepath.suffix.lower()
    meta = {
        "name":      filepath.stem,
        "filename":  filepath.name,
        "category":  classify_document(filepath.name, str(filepath)),
        "ext":       ext,
        "size_kb":   round(stat.st_size / 1024, 1),
        "modified":  datetime.fromtimestamp(stat.st_mtime).strftime("%d %b %Y"),
        "filepath":  str(filepath),
        "preview":   "",
        "full_text": "",
        "pages":     None,
    }

    full = _extract_full_text(filepath)
    if full:
        meta["full_text"] = full
        meta["preview"] = _extract_preview(full)

    if ext == ".pdf":
        try:
            import pypdf
            meta["pages"] = len(pypdf.PdfReader(str(filepath)).pages)
        except Exception:
            pass

    # xlsx: no text extraction
    return meta


def extract_folder_group(folder: Path) -> dict | None:
    """
    Treat a folder as a single logical document group.
    The folder name drives classification (same keyword rules as files).
    Every supported file inside becomes a file entry in the group.
    Nested subfolders are ignored.
    Returns None if the folder contains no supported files.
    """
    # Collect all supported files recursively within the folder
    files = sorted(
        f for f in folder.rglob("*")
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS
    )
    if not files:
        return None

    file_entries = []
    preview       = ""
    full_text_parts: list[str] = []
    most_recent   = ""

    for f in files:
        meta = extract_doc_metadata(f)
        ver  = _extract_version(f.stem)
        ver_label = ""
        if ver != (0, 0):
            major, minor = ver
            ver_label = f"Rev {major}" + (f".{minor}" if minor else "")

        file_entries.append({
            "ext":       meta["ext"],
            "filename":  meta["filename"],
            "filepath":  meta["filepath"],
            "size_kb":   meta["size_kb"],
            "pages":     meta["pages"],
            "ver_tuple": ver,
            "ver_label": ver_label,
            "modified":  meta["modified"],
        })
        if not preview and meta["preview"]:
            preview = meta["preview"]
        # Accumulate full text from every file in the folder
        if meta.get("full_text"):
            full_text_parts.append(f"[{meta['filename']}]\n{meta['full_text']}")
        if meta["modified"] > most_recent:
            most_recent = meta["modified"]

    # Sort: newest revision last, within same revision PDF → Word → Excel → PPT
    ext_order = {".pdf": 0, ".docx": 1, ".xlsx": 2, ".xls": 2, ".pptx": 3, ".ppt": 3}
    file_entries.sort(key=lambda f: (tuple(f["ver_tuple"]), ext_order.get(f["ext"], 9)))

    has_versions = len({tuple(f["ver_tuple"]) for f in file_entries}) > 1

    return {
        "title":        folder.name.replace("_", " ").replace("-", " "),
        "modified":     most_recent,
        "preview":      preview,
        "full_text":    "\n\n".join(full_text_parts),
        "has_versions": has_versions,
        "files":        file_entries,
        # Store category on the group itself for reclassification.
        # Pass folder path so a "Library/<Category>/My Study/" folder
        # inherits the parent category directly.
        "category":     classify_document(folder.name, str(folder)),
        "source":       "folder",
        "name":         folder.name,   # used by reclassification
        "filename":     folder.name,   # used by _has_type_keyword
    }


def _has_type_keyword(filename: str) -> bool:
    """Return True if the filename contains an explicit type keyword."""
    stem = Path(filename).stem
    return bool(
        re.search(r"protocol", stem, re.IGNORECASE)
        or re.search(r"report", stem, re.IGNORECASE)
        or re.search(r"(?<![A-Za-z])TR(?![A-Za-z])", stem, re.IGNORECASE)
        or re.search(r"(?<![A-Za-z])(ME|TM)(?![A-Za-z])", stem, re.IGNORECASE)
        or re.search(r"(?<![A-Za-z])WI(?![A-Za-z])", stem, re.IGNORECASE)
    )


# Categories that reclassification is allowed to assign to keyword-less files.
# "Study Reports" is deliberately excluded: a file with no keywords defaults
# to Study Protocols, and a similar-named report from the same study should
# NOT pull it into Study Reports — they are genuinely different documents.
# Only WI/TM/ME revision files (e.g. "AGN-2021-WI-001 Rev2") should inherit
# a non-default category, and those always have the keyword in a sibling file.
_RECLASSIFY_ALLOWED = {"Test Methods", "Work Instructions"}


def _reclassify_ambiguous(all_docs: list[dict]) -> list[dict]:
    """
    Second-pass classifier for files with no type keyword.

    Only promotes files into Test Methods or Work Instructions — never into
    Study Reports. This prevents protocols (which look identical to their
    corresponding reports except for the missing "Report"/"TR" keyword) from
    being incorrectly pulled into Study Reports.

    Use case this is designed for: revision files like
    "AGN-2021-WI-001 Luminex Rev2.docx" that lack "WI" in the filename
    but clearly belong with the WI they are a revision of.
    """
    from difflib import SequenceMatcher

    # Only consider confident docs whose category is one we're allowed to inherit
    confident = [
        d for d in all_docs
        if _has_type_keyword(d.get("filename", ""))
        and d.get("category") in _RECLASSIFY_ALLOWED
    ]
    ambiguous = [d for d in all_docs if not _has_type_keyword(d.get("filename", ""))]

    if not confident or not ambiguous:
        return all_docs

    conf_stems = [_clean_stem(d.get("name") or d.get("title", "")) for d in confident]

    for doc in ambiguous:
        doc_stem = _clean_stem(doc.get("name") or doc.get("title", ""))
        best_score, best_cat = 0.0, None
        for conf_doc, conf_stem in zip(confident, conf_stems):
            score = SequenceMatcher(None, doc_stem, conf_stem).ratio()
            if score > best_score:
                best_score = score
                best_cat   = conf_doc["category"]

        if best_score >= 0.70 and best_cat is not None:
            doc["category"] = best_cat

    return all_docs


def _collect_items(library_path: Path) -> tuple[list[Path], list[Path]]:
    """
    Recursively walk the library folder and return:
      - loose_files: files that sit directly in a folder with no siblings
        of the same document group, OR files at the root level
      - folder_items: directories at any depth whose contents should be
        treated as one logical document group

    Strategy:
      - Root-level files → loose files
      - Any directory → treated as a folder group (contents flattened)
      - Files inside a directory → handled by extract_folder_group, not here
    We walk only two levels: root files + immediate subdirs (which may
    themselves contain subdirs — extract_folder_group handles that recursively).
    """
    loose_files = []
    folder_items = []
    skip_names = {"library_index.json", "library_index.json.tmp"}
    # Category-named subfolders (Library/Study Protocols/ etc.) are not
    # document groups themselves — scan them like the root so their contents
    # get classified the normal way.
    category_names = {k.lower() for k in LIBRARY_CATEGORIES}

    def _walk(current_dir: Path) -> None:
        try:
            entries = sorted(current_dir.iterdir())
        except PermissionError:
            return

        for item in entries:
            if item.name.startswith(".") or item.name in skip_names:
                continue
            if item.is_file():
                if item.suffix.lower() in SUPPORTED_EXTS:
                    loose_files.append(item)
            elif item.is_dir():
                if item.name.lower() in category_names:
                    _walk(item)  # pass-through; treat contents as root-level
                else:
                    folder_items.append(item)

    _walk(library_path)
    return loose_files, folder_items


def _scan_library_raw(library_path: str) -> dict[str, list[dict]]:
    """
    Raw scan — classifies and groups all documents in the library.
    Handles nested folder structures by treating each subfolder as one
    logical document group regardless of nesting depth.
    """
    path = Path(library_path)
    if not path.exists():
        return {k: [] for k in LIBRARY_CATEGORIES}

    loose_files, folder_items = _collect_items(path)

    total = len(loose_files) + len(folder_items)
    done = 0
    progress = _scan_progress_callback or (lambda d, t, m: None)

    loose_docs: list[dict] = []
    for f in loose_files:
        progress(done, total, f"Reading {f.name}")
        try:
            loose_docs.append(_with_timeout(extract_doc_metadata, (f,), 30))
        except TimeoutError:
            print(f"Warning: timed out reading {f.name}")
        except Exception as e:
            # Log and skip files that can't be read (locked, corrupt, etc.)
            print(f"Warning: could not read {f.name}: {e}")
        done += 1

    folder_groups: list[dict] = []
    for folder in folder_items:
        progress(done, total, f"Scanning folder {folder.name}")
        try:
            grp = _with_timeout(extract_folder_group, (folder,), 60)
            if grp:
                folder_groups.append(grp)
        except TimeoutError:
            print(f"Warning: timed out scanning folder {folder.name}")
        except Exception as e:
            print(f"Warning: could not process folder {folder.name}: {e}")
        done += 1

    progress(total, total, "Classifying…")

    loose_docs = _reclassify_ambiguous(loose_docs)

    # Reclassify folder groups using confident loose docs as anchors
    confident_loose = [
        {"name": d.get("name",""), "filename": d.get("filename",""), "category": d.get("category","Study Protocols")}
        for d in loose_docs if _has_type_keyword(d.get("filename",""))
    ]
    folder_proxies = [
        {"name": g.get("name",""), "filename": g.get("filename",""), "category": g.get("category","Study Protocols")}
        for g in folder_groups
    ]
    reclassified = _reclassify_ambiguous(folder_proxies + confident_loose)
    for grp, proxy in zip(folder_groups, reclassified[:len(folder_groups)]):
        grp["category"] = proxy["category"]

    categorised: dict[str, list[dict]] = {k: [] for k in LIBRARY_CATEGORIES}
    for doc in loose_docs:
        categorised[doc["category"]].append(doc)
    for grp in folder_groups:
        cat = grp.get("category", "Study Protocols")
        clean_grp = {k: v for k, v in grp.items()
                     if k not in ("category", "source", "name", "filename")}
        categorised[cat].append({"_folder_group": True, **clean_grp})

    return categorised


def _index_to_categorised(index: dict) -> dict[str, list[dict]]:
    """Convert a raw index dict into the categorised groups dict the UI expects."""
    categorised: dict[str, list[dict]] = {k: [] for k in LIBRARY_CATEGORIES}
    for entry in index.values():
        if entry.get("deleted"):
            continue
        cat = entry.get("category", "Study Protocols")
        if cat not in categorised:
            cat = "Study Protocols"
        group = {
            "title":         entry.get("display_name") or entry.get("title", ""),
            "modified":      entry.get("modified", ""),
            "preview":       entry.get("preview", ""),
            "has_versions":  entry.get("has_versions", False),
            "files":         entry.get("files", []),
            "_folder_group": entry.get("source") == "folder",
            "_index_id":     entry["id"],
        }
        categorised[cat].append(group)
    for cat in categorised:
        categorised[cat].sort(key=lambda g: g["title"].lower())
    return categorised


def _normalise_entry(
    doc: dict,
    category: str,
    existing_ids: set[str],
    now: str,
) -> tuple[str, dict]:
    """
    Convert any doc/group shape into a well-formed index entry. Handles:
      - Folder groups: already have "files", "title", "has_versions"
      - Loose file docs: flat dict with "filepath", "name", "ext" etc.
    `existing_ids` is used for dedup-by-suffix so that two loose files with
    the same stem get unique ids.
    """
    is_folder_group = doc.get("_folder_group", False)

    if is_folder_group:
        files_raw = doc.get("files", [])
        title     = doc.get("title", "")
        has_ver   = doc.get("has_versions", False)
        preview   = doc.get("preview", "")
        modified  = doc.get("modified", "")
        source    = "folder"
    else:
        ver = _extract_version(doc.get("name", ""))
        files_raw = [{
            "ext":       doc.get("ext", ""),
            "filename":  doc.get("filename", ""),
            "filepath":  doc.get("filepath", ""),
            "size_kb":   doc.get("size_kb", 0),
            "pages":     doc.get("pages"),
            "ver_tuple": list(ver),
            "ver_label": "",
            "modified":  doc.get("modified", ""),
        }]
        title    = doc.get("name", doc.get("filename", "unknown"))
        has_ver  = False
        preview  = doc.get("preview", "")
        modified = doc.get("modified", "")
        source   = "file"

    files_with_mtime = []
    for f in files_raw:
        fc = dict(f)
        try:
            fc["_mtime_iso"] = datetime.fromtimestamp(
                Path(fc["filepath"]).stat().st_mtime
            ).isoformat()
        except (OSError, KeyError):
            fc["_mtime_iso"] = ""
        if isinstance(fc.get("ver_tuple"), tuple):
            fc["ver_tuple"] = list(fc["ver_tuple"])
        files_with_mtime.append(fc)

    if files_with_mtime:
        first = files_with_mtime[0]
        fp = Path(first.get("filepath", ""))
        if source == "folder":
            entry_id = fp.parent.name
        else:
            entry_id = fp.stem + "_" + fp.suffix.lstrip(".")
    else:
        entry_id = title.replace(" ", "_")[:60]

    base_id = entry_id
    counter = 1
    while entry_id in existing_ids:
        entry_id = f"{base_id}_{counter}"
        counter += 1

    entry = {
        "id":              entry_id,
        "title":           title,
        "auto_category":   category,
        "manual_category": None,
        "category":        category,
        "files":           files_with_mtime,
        "preview":         preview,
        "modified":        modified,
        "has_versions":    has_ver,
        "source":          source,
        "display_name":    None,
        "deleted":         False,
        "last_indexed":    now,
    }
    return entry_id, entry


def _build_index_from_scan(library_path: str) -> dict:
    """
    Run a full raw scan and convert results into an index dict.
    Each entry in the index represents one document group.
    """
    import json, os
    from datetime import datetime

    categorised = _scan_library_raw(library_path)
    index = {}
    now = datetime.now().isoformat()

    for category, docs in categorised.items():
        for doc in docs:
            entry_id, entry = _normalise_entry(doc, category, set(index.keys()), now)
            index[entry_id] = entry

    # Write index to disk with retry (os.replace unreliable on Windows network drives)
    index_path = Path(library_path) / "library_index.json"
    payload = {
        "version":      2,
        "last_updated": now,
        "entries":      list(index.values()),
    }
    import time as _time
    for _attempt in range(5):
        try:
            with open(str(index_path), "w", encoding="utf-8") as _fh:
                json.dump(payload, _fh, indent=2)
            break
        except PermissionError:
            if _attempt < 4:
                _time.sleep(0.5)
            else:
                print("Warning: could not write library index after 5 attempts.")

    return index


def _load_index_from_disk(library_path: str) -> dict:
    """Read library_index.json and return as a dict keyed by entry id."""
    import json
    index_path = Path(library_path) / "library_index.json"
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {e["id"]: e for e in data.get("entries", [])}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _check_for_changes(library_path: str, index: dict) -> bool:
    """
    Quick check: return True if any files on disk differ from the index.
    Checks file mtimes and looks for new files not in the index.
    """
    indexed_paths = set()
    for entry in index.values():
        for f in entry.get("files", []):
            indexed_paths.add(f.get("filepath", ""))
            stored_mtime = f.get("_mtime_iso", "")
            try:
                from datetime import datetime as dt
                actual_mtime = dt.fromtimestamp(
                    Path(f["filepath"]).stat().st_mtime
                ).isoformat()
                if actual_mtime != stored_mtime:
                    return True
            except OSError:
                return True  # file gone

    # Check for new files not in index
    path = Path(library_path)
    for item in path.iterdir():
        if item.name == "library_index.json" or item.name.startswith("."):
            continue
        if item.is_file() and item.suffix.lower() in SUPPORTED_EXTS:
            if str(item) not in indexed_paths:
                return True
        elif item.is_dir():
            for f in item.iterdir():
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS:
                    if str(f) not in indexed_paths:
                        return True
    return False


def load_library_from_index(library_path: str) -> dict[str, list[dict]]:
    """
    Fast library load using a two-level cache:
      1. st.session_state  — zero I/O on rerenders within the same session
      2. library_index.json on disk — one JSON read per session start
      3. Full raw scan — only when no index exists yet.
    Incremental refresh is triggered manually via the Library tab's Sync button.
    """
    # ── Level 1: in-memory session cache ────────────────────────────────────
    if "library_cache" in st.session_state:
        return st.session_state["library_cache"]

    index_path = Path(library_path) / "library_index.json"

    # ── Level 2: read existing index ────────────────────────────────────────
    if index_path.exists():
        index = _load_index_from_disk(library_path)

    # ── Level 3: first run — build index now ────────────────────────────────
    else:
        global _scan_progress_callback
        st.info("No library index found — building one now. This runs once; future launches are instant.")
        bar = st.progress(0, text="Starting first-time library index build…")
        def _cb(done: int, total: int, msg: str) -> None:
            pct = int(done / max(total, 1) * 100)
            bar.progress(min(pct, 100), text=f"{msg} ({done}/{total})")
        _scan_progress_callback = _cb
        try:
            index = _build_index_from_scan(library_path)
        finally:
            _scan_progress_callback = None
        bar.empty()

    result = _normalise_ver_tuples(_index_to_categorised(index))
    st.session_state["library_cache"] = result
    return result


def _normalise_ver_tuples(categorised: dict) -> dict:
    """
    Ensure all ver_tuple values are Python tuples (not lists).
    JSON deserialisation converts tuples to lists; this fixes them on load.
    """
    for groups in categorised.values():
        for group in groups:
            for f in group.get("files", []):
                if isinstance(f.get("ver_tuple"), list):
                    f["ver_tuple"] = tuple(f["ver_tuple"])
    return categorised


def load_library_from_index_clear() -> None:
    """Clear session cache so the next load re-reads the index from disk."""
    st.session_state.pop("library_cache", None)


def load_library_rebuild() -> None:
    """Delete the index file and clear session cache — forces a full rescan."""
    import os
    st.session_state.pop("library_cache", None)
    index_path = Path(LIBRARY_PATH) / "library_index.json"
    try:
        os.remove(str(index_path))
    except FileNotFoundError:
        pass


def _render_organize_section() -> None:
    """
    Library organize tool. One-click migration that moves root-level items
    into category subfolders based on the current classifier, letting the
    user adjust individual rows before committing. After apply, future
    classification works purely off folder location.
    """
    plan = _build_organize_plan()

    # Hide the button entirely when the library root is already tidy
    if not plan:
        lib_path = Path(LIBRARY_PATH)
        if any((lib_path / cat).exists() for cat in LIBRARY_CATEGORIES):
            return  # already organized, no need to show anything
        # Library has no category folders yet and nothing to move (empty lib)
        return

    with st.expander(
        f"🗂 Organize library — {len(plan)} item(s) at root could move into "
        "category folders",
        expanded=False,
    ):
        st.caption(
            "Auto-sort your existing library into `Study Protocols/`, "
            "`Study Reports/`, `Test Methods/`, and `Work Instructions/` "
            "subfolders. Review and adjust each destination below. "
            "Moves are performed on disk."
        )

        selections: dict[str, str] = {}
        choices = list(LIBRARY_CATEGORIES.keys()) + ["(skip)"]

        # Header row
        hc1, hc2, hc3 = st.columns([5, 1, 3])
        with hc1: st.caption("**Item**")
        with hc2: st.caption("")
        with hc3: st.caption("**Destination**")

        for row in plan:
            sp = str(row["path"])
            c_name, c_arrow, c_cat = st.columns([5, 1, 3])
            with c_name:
                icon = "📁" if row["is_dir"] else "📄"
                st.markdown(f"{icon} `{row['path'].name}`")
            with c_arrow:
                st.markdown("→")
            with c_cat:
                default = row["category"]
                chosen = st.selectbox(
                    "Destination",
                    choices,
                    index=choices.index(default) if default in choices else 0,
                    key=f"organize_dest_{sp}",
                    label_visibility="collapsed",
                )
                selections[sp] = chosen

        st.markdown(
            "<div style='margin-top:0.5rem'></div>",
            unsafe_allow_html=True,
        )
        col_apply, col_skip_all = st.columns([2, 1])
        with col_apply:
            if st.button("✅ Apply moves", type="primary",
                         use_container_width=True, key="organize_apply"):
                moved, errors = _apply_organize_plan(selections)
                for err in errors:
                    st.warning(err)
                if moved:
                    st.success(f"Moved {moved} item(s). Rescanning…")
                    _run_scan_with_progress()
                else:
                    st.info("Nothing moved.")
                st.rerun()
        with col_skip_all:
            st.caption(
                "_Tip: set a row to **(skip)** to leave that item at root._"
            )


def _build_organize_plan() -> list[dict]:
    """
    Inspect the library root and return a move plan: for each item that lives
    at the root (not already inside a category subfolder), propose a
    destination category based on filename/folder-name keywords.
    """
    lib_path = Path(LIBRARY_PATH)
    if not lib_path.exists():
        return []

    category_names_lower = {k.lower() for k in LIBRARY_CATEGORIES}
    skip_names = {"library_index.json", "library_index.json.tmp",
                  "library_vectors.npz"}

    plan: list[dict] = []
    try:
        for item in sorted(lib_path.iterdir()):
            if item.name.startswith(".") or item.name in skip_names:
                continue
            if item.is_dir() and item.name.lower() in category_names_lower:
                continue  # already a category folder
            if item.is_file() and item.suffix.lower() not in SUPPORTED_EXTS:
                continue  # unsupported filetype (.xlsx workbook, etc.)
            plan.append({
                "path":     item,
                "is_dir":   item.is_dir(),
                "category": classify_document(item.name, str(item)),
            })
    except Exception as e:
        print(f"organize: enumerate failed: {e}")
        return []
    return plan


def _apply_organize_plan(selections: dict[str, str]) -> tuple[int, list[str]]:
    """
    Execute the user-approved move plan. Creates category subfolders as
    needed. Returns (moves_applied, errors).
    """
    import shutil
    lib_path = Path(LIBRARY_PATH).resolve()
    for cat in LIBRARY_CATEGORIES:
        (lib_path / cat).mkdir(exist_ok=True)

    moved = 0
    errors: list[str] = []
    for src_str, dest_cat in selections.items():
        if dest_cat == "(skip)":
            continue
        src = Path(src_str).resolve()
        # Guard against path traversal: only move files that live under the
        # library root. UI keys are the plan dict's own keys today, but an
        # adversarial plan dict (or a future code path) could include paths
        # pointing elsewhere on disk.
        try:
            src.relative_to(lib_path)
        except ValueError:
            errors.append(f"{src.name}: refusing to move file outside library root")
            continue
        if not src.exists():
            continue
        dst = lib_path / dest_cat / src.name
        if dst.exists():
            errors.append(
                f"{src.name}: '{dest_cat}/{src.name}' already exists — skipped"
            )
            continue
        try:
            shutil.move(str(src), str(dst))
            moved += 1
        except Exception as e:
            errors.append(f"{src.name}: {e}")
    return moved, errors


def _incremental_scan_update() -> dict:
    """
    Fast sync — only processes files that are new, modified, or deleted
    since the last scan. Preserves existing entries (and their manual
    overrides: category, display_name, deleted flag) for unchanged files.

    If no existing index exists, falls through to a full rebuild.
    """
    import json as _json
    import os
    from datetime import datetime as _dt

    index_path = Path(LIBRARY_PATH) / "library_index.json"
    if not index_path.exists():
        return _run_scan_with_progress()

    try:
        with open(index_path, "r", encoding="utf-8") as f:
            existing_data = _json.load(f)
    except Exception:
        return _run_scan_with_progress()

    existing_by_id: dict[str, dict] = {
        e["id"]: e for e in existing_data.get("entries", [])
    }

    # Build filepath -> mtime lookup from existing entries
    existing_file_mtimes: dict[str, str] = {}
    for entry in existing_by_id.values():
        for f in entry.get("files", []):
            existing_file_mtimes[f.get("filepath", "")] = f.get("_mtime_iso", "")

    def _iso(p: Path) -> str:
        try:
            return _dt.fromtimestamp(p.stat().st_mtime).isoformat()
        except OSError:
            return ""

    # Snapshot current disk state
    loose_files, folder_items = _collect_items(Path(LIBRARY_PATH))

    # What's new or modified?
    changed_loose = [
        f for f in loose_files
        if existing_file_mtimes.get(str(f), "") != _iso(f)
    ]

    def _folder_changed(folder: Path) -> bool:
        try:
            for inner in folder.rglob("*"):
                if inner.is_file() and inner.suffix.lower() in SUPPORTED_EXTS:
                    if existing_file_mtimes.get(str(inner), "") != _iso(inner):
                        return True
        except OSError:
            pass
        return False

    changed_folders = [f for f in folder_items if _folder_changed(f)]

    # What's been deleted? (entries whose files are all gone)
    current_filepaths: set[str] = {str(f) for f in loose_files}
    for folder in folder_items:
        try:
            for inner in folder.rglob("*"):
                if inner.is_file() and inner.suffix.lower() in SUPPORTED_EXTS:
                    current_filepaths.add(str(inner))
        except OSError:
            pass
    deleted_ids: list[str] = []
    for eid, entry in existing_by_id.items():
        entry_files = {f.get("filepath", "") for f in entry.get("files", [])}
        if entry_files and not (entry_files & current_filepaths):
            deleted_ids.append(eid)

    total = len(changed_loose) + len(changed_folders) + len(deleted_ids)
    if total == 0:
        st.info("Library is already up to date — no changes detected.")
        return existing_data

    bar = st.progress(0, text=f"Syncing {total} change(s)…")
    done = 0

    # Remove deleted entries
    for eid in deleted_ids:
        existing_by_id.pop(eid, None)
    done += len(deleted_ids)
    bar.progress(int(done / total * 100))

    # Reprocess changed loose files — first drop their old entries so ids
    # recompute cleanly and reclassification can re-apply
    def _loose_eid(fp: Path) -> str:
        return fp.stem + "_" + fp.suffix.lstrip(".")
    for f in changed_loose:
        existing_by_id.pop(_loose_eid(f), None)
    for folder in changed_folders:
        existing_by_id.pop(folder.name, None)

    new_loose_docs: list[dict] = []
    for f in changed_loose:
        bar.progress(
            int(done / total * 100),
            text=f"Reading {f.name} ({done + 1}/{total})",
        )
        try:
            new_loose_docs.append(_with_timeout(extract_doc_metadata, (f,), 30))
        except TimeoutError:
            print(f"Warning: timed out reading {f.name}")
        except Exception as e:
            print(f"Warning: could not read {f.name}: {e}")
        done += 1

    # Apply the sibling-based reclassifier so a protocol revision drops into
    # the right bucket even without the "Protocol" keyword
    new_loose_docs = _reclassify_ambiguous(new_loose_docs)

    new_folder_groups: list[dict] = []
    for folder in changed_folders:
        bar.progress(
            int(done / total * 100),
            text=f"Scanning {folder.name} ({done + 1}/{total})",
        )
        try:
            grp = _with_timeout(extract_folder_group, (folder,), 60)
            if grp:
                new_folder_groups.append(grp)
        except TimeoutError:
            print(f"Warning: timed out scanning folder {folder.name}")
        except Exception as e:
            print(f"Warning: could not scan folder {folder.name}: {e}")
        done += 1

    # Merge new entries into the existing id map
    now = _dt.now().isoformat()
    for doc in new_loose_docs:
        cat = doc.get("category", "Study Protocols")
        eid, entry = _normalise_entry(
            doc, cat, set(existing_by_id.keys()), now
        )
        existing_by_id[eid] = entry
    for grp in new_folder_groups:
        cat = grp.get("category", "Study Protocols")
        grp_copy = {k: v for k, v in grp.items()
                    if k not in ("category", "source", "name", "filename")}
        grp_copy["_folder_group"] = True
        eid, entry = _normalise_entry(
            grp_copy, cat, set(existing_by_id.keys()), now
        )
        existing_by_id[eid] = entry

    # Persist merged index
    merged = {
        "version":      2,
        "last_updated": now,
        "entries":      list(existing_by_id.values()),
    }
    tmp_path = str(index_path) + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        _json.dump(merged, f, indent=2)
    os.replace(tmp_path, str(index_path))

    bar.empty()
    st.session_state.pop("library_cache", None)

    added = len(new_loose_docs) + len(new_folder_groups)
    parts = []
    if added:
        parts.append(f"{added} added/updated")
    if deleted_ids:
        parts.append(f"{len(deleted_ids)} removed")
    st.success("Sync complete — " + ", ".join(parts) + ".")
    return merged


def _run_scan_with_progress() -> dict:
    """
    Run a full library rescan with a Streamlit progress bar and per-file
    timeout so a single bad PDF can't stall the whole scan. Returns the
    fresh index dict.
    """
    import os
    global _scan_progress_callback

    st.session_state.pop("library_cache", None)
    index_path = Path(LIBRARY_PATH) / "library_index.json"
    try:
        os.remove(str(index_path))
    except FileNotFoundError:
        pass

    bar = st.progress(0, text="Starting scan…")
    def _cb(done: int, total: int, msg: str) -> None:
        pct = int(done / max(total, 1) * 100)
        bar.progress(min(pct, 100), text=f"{msg} ({done}/{total})")

    _scan_progress_callback = _cb
    try:
        new_index = _build_index_from_scan(LIBRARY_PATH)
    finally:
        _scan_progress_callback = None
    bar.empty()
    st.success(f"Scan complete — {len(new_index)} document groups.")
    return new_index


# Keep old name as alias
def load_library(library_path: str) -> dict[str, list[dict]]:
    return load_library_from_index(library_path)


_clean_stem = _shared_clean_stem

def _extract_version(stem: str) -> tuple[int, int]:
    """
    Extract a (major, minor) version tuple for sorting within a group.
    Looks for patterns like v2, Rev3, Revision 2, r4, _final, _approved.
    Files with no version marker get (0, 0) so they sort first as the
    original, and later revisions sort higher.
    "final", "approved", "signed" are treated as (999, 0) — always latest.
    """
    s = stem.lower()

    # "revision 3" or "rev3" or "r3" at end of string
    m = re.search(r'(?:revision|rev|r)[\s_\-]*(\d+)', s)
    if m:
        return (int(m.group(1)), 0)

    # "v2" or "v2.1"
    m = re.search(r'v(\d+)(?:[._](\d+))?', s)
    if m:
        major = int(m.group(1))
        minor = int(m.group(2)) if m.group(2) else 0
        return (major, minor)

    return (0, 0)


def _similarity(a: str, b: str) -> float:
    """SequenceMatcher ratio between two cleaned stems."""
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a, b).ratio()


def group_documents(docs: list[dict]) -> list[dict]:
    """
    Group documents that belong together — same document in different formats
    or multiple revisions of the same document.

    Folder groups (marked _folder_group=True) pass straight through.
    Loose files are grouped using union-find with a similarity threshold.
    """
    # Separate pre-built folder groups from loose files
    folder_groups = [d for d in docs if d.get("_folder_group")]
    loose_docs    = [d for d in docs if not d.get("_folder_group")]

    THRESHOLD = 0.92

    n = len(loose_docs)
    cleaned = [_clean_stem(d.get("name") or d.get("title", "")) for d in loose_docs]

    # Union-Find
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    # Block by first token (usually the study number) so SequenceMatcher only
    # runs between plausible matches. For the full library scan this turns a
    # several-thousand-comparison pass into at most a few hundred.
    from collections import defaultdict as _dd
    blocks: dict[str, list[int]] = _dd(list)
    for i, s in enumerate(cleaned):
        first = s.split(" ", 1)[0] if s else ""
        blocks[first].append(i)

    for idxs in blocks.values():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                i, j = idxs[a], idxs[b]
                if _similarity(cleaned[i], cleaned[j]) >= THRESHOLD:
                    union(i, j)

    # Collect groups
    from collections import defaultdict
    buckets: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        buckets[find(i)].append(i)

    groups = []
    for indices in buckets.values():
        members = [loose_docs[i] for i in indices]

        # Build file entries with version labels
        # Members may be raw scan dicts (have "name") or index-reconstructed
        # group dicts (have "title" and a pre-built "files" list). Handle both.
        file_entries = []
        for doc in members:
            stem = doc.get("name") or doc.get("title", "")
            if "files" in doc and not doc.get("ext"):
                # Pre-built group from index — expand its files directly
                for f in doc["files"]:
                    file_entries.append(f)
                continue
            ver = _extract_version(stem)
            if ver == (0, 0):
                ver_label = ""
            else:
                major, minor = ver
                ver_label = f"Rev {major}" + (f".{minor}" if minor else "")
            file_entries.append({
                "ext":         doc.get("ext", ""),
                "filename":    doc.get("filename", ""),
                "filepath":    doc.get("filepath", ""),
                "size_kb":     doc.get("size_kb", 0),
                "pages":       doc.get("pages"),
                "ver_tuple":   ver,
                "ver_label":   ver_label,
                "modified":    doc.get("modified", ""),
            })

        # Skip groups that ended up with no files (stale index entries)
        if not file_entries:
            continue

        # Sort: older revisions first, within same revision PDF before Word
        file_entries.sort(key=lambda f: (tuple(f["ver_tuple"]), 0 if f["ext"] == ".pdf" else 1))

        # Display title: use the filename of the latest revision with version
        # suffixes stripped and separators normalised.
        latest = max(file_entries, key=lambda f: tuple(f["ver_tuple"]))
        title = display_title_from_filename(latest["filename"])

        # Most recent modified date
        most_recent = max(f["modified"] for f in file_entries)
        preview = next((d.get("preview","") for d in members if d.get("preview")), "")
        has_versions = len({tuple(f["ver_tuple"]) for f in file_entries}) > 1

        groups.append({
            "title":        title,
            "modified":     most_recent,
            "preview":      preview,
            "has_versions": has_versions,
            "files":        file_entries,
        })

    # Merge folder groups and loose file groups, sort all alphabetically
    all_groups = folder_groups + groups
    all_groups.sort(key=lambda g: g["title"].lower())
    return all_groups



def _download_btn(file: dict, key: str) -> None:
    """One-click download button."""
    ext_info = SUPPORTED_EXTS.get(
        file["ext"],
        {"label": file["ext"].lstrip(".").upper(), "mime": "application/octet-stream"}
    )
    try:
        with open(file["filepath"], "rb") as fh:
            st.download_button(
                label=f"\u2b07 {ext_info['label']}",
                data=fh.read(),
                file_name=file["filename"],
                mime=ext_info["mime"],
                key=key,
                use_container_width=True,
            )
    except OSError:
        st.caption("\u26a0 unavailable")


EXT_ICONS = {
    ".pdf":  "🔴",
    ".docx": "🔵",
    ".doc":  "🔵",
    ".xlsx": "🟢",
    ".xls":  "🟢",
    ".pptx": "🟠",
    ".ppt":  "🟠",
}


def _file_display_name(filename: str) -> str:
    """
    Return a clean human-readable name for a file.
    Strips the extension and replaces separators with spaces.
    e.g. "AB-001_Measurement_Form_v2.xlsx" → "AB-001 Measurement Form v2"
    """
    stem = Path(filename).stem
    return re.sub(r'[\s_\-]+', ' ', stem).strip()


def _render_file_row(f: dict, key: str, show_name: bool = True) -> None:
    """
    Render a single file as a row:
      🔴 Filename.pdf   42 KB · 8p        ⬇ PDF
    """
    icon      = EXT_ICONS.get(f["ext"], "📄")
    ext_info  = SUPPORTED_EXTS.get(f["ext"], {"label": f["ext"].lstrip(".").upper()})
    disp_name = _file_display_name(f["filename"]) if show_name else ext_info["label"]
    size_str  = f"{f['size_kb']} KB"
    if f.get("pages"):
        size_str += f" · {f['pages']}p"

    col_name, col_size, col_btn = st.columns([5, 2, 1])
    with col_name:
        st.markdown(
            f"<span style='font-size:13px'>{icon} {disp_name}</span>",
            unsafe_allow_html=True,
        )
    with col_size:
        st.caption(size_str)
    with col_btn:
        _download_btn(f, key=key)


def render_doc_card(group: dict, card_key: str) -> None:
    """
    Render one document card.

    Every card shows:
      - Title + modified date
      - A row per file with: icon, filename, size, download button
      - Preview expander (if available) pulled from the first readable file

    Multi-revision groups additionally show a revision badge and group
    files under their revision label.
    """
    is_multi_file    = len(group["files"]) > 1
    has_versions     = group["has_versions"]
    n_versions       = len({tuple(f["ver_tuple"]) for f in group["files"]}) if has_versions else 0

    with st.container(border=True):
        # ── Title row ────────────────────────────────────────────────────
        title_md = f"**{group['title']}**"
        if has_versions:
            title_md += (
                f" &nbsp;<span style='font-size:11px;"
                f"background:var(--color-background-info);"
                f"color:var(--color-text-info);padding:2px 7px;"
                f"border-radius:10px'>🔄 {n_versions} revisions</span>"
            )
        st.markdown(title_md, unsafe_allow_html=True)
        st.caption(f"📅 {group['modified']}")

        # ── Preview ──────────────────────────────────────────────────────
        if group["preview"]:
            with st.expander("Preview", expanded=False):
                st.markdown(
                    f"<p style='font-size:12px;line-height:1.6;"
                    f"color:var(--color-text-secondary)'>"
                    f"{group['preview'][:350]}…</p>",
                    unsafe_allow_html=True,
                )

        # ── File list ────────────────────────────────────────────────────
        if not has_versions:
            # Simple list: every file gets a named row
            st.markdown(
                "<hr style='margin:6px 0;border:none;"
                "border-top:0.5px solid var(--color-border-tertiary)'>",
                unsafe_allow_html=True,
            )
            for fi, f in enumerate(group["files"]):
                # Only show filename if there are multiple files OR the
                # filename differs meaningfully from the card title
                show_name = is_multi_file or (
                    _clean_stem(Path(f["filename"]).stem).lower()
                    != _clean_stem(group["title"]).lower()
                )
                _render_file_row(f, key=f"{card_key}_f{fi}", show_name=show_name)

        else:
            # Grouped by revision, newest first
            st.markdown(
                "<hr style='margin:6px 0;border:none;"
                "border-top:0.5px solid var(--color-border-tertiary)'>",
                unsafe_allow_html=True,
            )
            ver_groups: dict = {}
            for f in group["files"]:
                ver_groups.setdefault(tuple(f["ver_tuple"]), []).append(f)

            for vi, (ver_tuple, ver_files) in enumerate(
                sorted(ver_groups.items(), key=lambda x: tuple(x[0]), reverse=True)
            ):
                ver_label = ver_files[0]["ver_label"] or "Original"
                st.markdown(
                    f"<span style='font-size:11px;font-weight:500;"
                    f"color:var(--color-text-secondary)'>{ver_label}</span>",
                    unsafe_allow_html=True,
                )
                sorted_ver_files = sorted(
                    ver_files,
                    key=lambda x: list(SUPPORTED_EXTS.keys()).index(x["ext"])
                    if x["ext"] in SUPPORTED_EXTS else 99
                )
                for bi, f in enumerate(sorted_ver_files):
                    _render_file_row(f, key=f"{card_key}_v{vi}_b{bi}", show_name=True)



def render_library() -> None:
    """Render the full document library UI."""

    # ── Header row ──────────────────────────────────────────────────────────
    h_col, sync_col, rebuild_col = st.columns([4, 1, 1])
    with h_col:
        st.subheader("Document Library")
        st.caption(f"Library folder: `{LIBRARY_PATH}`")
    with sync_col:
        st.markdown("<div style='margin-top:1.6rem'></div>", unsafe_allow_html=True)
        if st.button("⚡ Sync", use_container_width=True,
                     help="Process only new, modified, or deleted files since the last scan. Fast — use this after adding/removing files."):
            _incremental_scan_update()
            st.rerun()
    with rebuild_col:
        st.markdown("<div style='margin-top:1.6rem'></div>", unsafe_allow_html=True)
        if st.button("↻ Rebuild", use_container_width=True,
                     help="Full rebuild from scratch. Slower — use only if the index is corrupted or classification rules changed."):
            _run_scan_with_progress()
            st.rerun()

    # ── Organize library tool ───────────────────────────────────────────────
    _render_organize_section()

    # ── Folder check ────────────────────────────────────────────────────────
    if not Path(LIBRARY_PATH).exists():
        st.warning(
            f"Library folder not found at `{LIBRARY_PATH}`. "
            "Create a folder called **Library** in the Testing directory "
            "and add your documents to it."
        )
        return

    # ── Raw disk diagnostic — only shown when index ≠ disk ───────────────────
    lib_path = Path(LIBRARY_PATH)
    top_files, top_dirs = [], []
    try:
        for item in sorted(lib_path.iterdir()):
            if item.name.startswith(".") or item.name == "library_index.json":
                continue
            if item.is_file():
                top_files.append(item)
            elif item.is_dir():
                top_dirs.append(item)
    except Exception as e:
        st.error(f"Could not read library folder: {e}")

    total_files_on_disk = sum(
        1 for f in top_files if f.suffix.lower() in SUPPORTED_EXTS
    )
    folder_file_counts: dict[str, int | str] = {}
    for d in top_dirs:
        try:
            count = sum(1 for f in d.rglob("*")
                       if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS)
            folder_file_counts[d.name] = count
            total_files_on_disk += count
        except Exception:
            folder_file_counts[d.name] = "?"

    index_path = lib_path / "library_index.json"
    index_entry_count = 0
    index_last_updated = ""
    if index_path.exists():
        try:
            import json as _json
            with open(index_path) as _f:
                _idx = _json.load(_f)
            entries_all = _idx.get("entries", [])
            index_entry_count = sum(1 for e in entries_all if not e.get("deleted"))
            index_last_updated = _idx.get("last_updated", "")
        except Exception:
            pass

    # A mismatch means the index is out of date — expand the diagnostic so the
    # user can see which folders are affected and force a rescan. When counts
    # match, stay silent.
    mismatch = index_path.exists() and total_files_on_disk != index_entry_count

    if mismatch:
        with st.expander(
            f"⚠️ Library mismatch — disk has {total_files_on_disk} files, "
            f"index has {index_entry_count}. Click to investigate.",
            expanded=True,
        ):
            st.caption(f"Files at root: {len([f for f in top_files if f.suffix.lower() in SUPPORTED_EXTS])}"
                       f"  |  Subfolders: {len(top_dirs)}")
            if folder_file_counts:
                st.markdown("**Files per subfolder:**")
                for name, count in sorted(folder_file_counts.items()):
                    st.caption(f"  📁 {name}: {count} files")
            if index_last_updated:
                st.caption(f"Index last updated: {index_last_updated}")

            if st.button("Force full rescan now", key="force_rescan"):
                _run_scan_with_progress()
                st.rerun()

    library = load_library(LIBRARY_PATH)
    total   = sum(len(v) for v in library.values())

    if total == 0:
        st.info("The library folder is empty. Add .docx or .pdf files to get started.")
        return

    # Diagnostic: show per-category counts so missing files are visible
    with st.expander(f"📊 Index summary ({total} document groups)", expanded=False):
        for cat, docs in library.items():
            if docs:
                st.caption(f"{CATEGORY_ICONS.get(cat,'')} {cat}: {len(docs)} groups")
        index_path = Path(LIBRARY_PATH) / "library_index.json"
        if index_path.exists():
            import json
            with open(index_path) as _f:
                _idx = json.load(_f)
            raw_count = len(_idx.get("entries", []))
            st.caption(f"Raw index entries: {raw_count}")
            st.caption(f"Index last updated: {_idx.get('last_updated','unknown')}")

    # ── Search — sits above tabs so it filters across the whole library ────
    search = st.text_input(
        "🔍 Search",
        placeholder="Search across all categories…",
        label_visibility="collapsed",
    )
    search_lower = search.strip().lower()

    # ── Group documents within each category ────────────────────────────────
    grouped_library = {cat: group_documents(docs) for cat, docs in library.items()}

    # ── Apply search filter across ALL categories before building tabs ───────
    # When a search is active the tab counts reflect filtered results so the
    # user can immediately see which categories have matches.
    def _matches(group: dict) -> bool:
        return (
            search_lower in group["title"].lower()
            or any(search_lower in f["filename"].lower() for f in group["files"])
            or search_lower in group.get("preview", "").lower()
        )

    if search_lower:
        filtered_library = {cat: [g for g in groups if _matches(g)]
                            for cat, groups in grouped_library.items()}
    else:
        filtered_library = grouped_library

    # ── Category tabs ───────────────────────────────────────────────────────
    # Always show all categories even if empty during a search, so the user
    # knows the search was applied everywhere.
    active_cats = [k for k, docs in library.items() if docs]

    tab_labels = [
        f"{CATEGORY_ICONS[k]}  {LIBRARY_CATEGORIES[k]}  ({len(filtered_library[k])})"
        for k in active_cats
    ]
    tabs = st.tabs(tab_labels)

    for tab, cat_key in zip(tabs, active_cats):
        with tab:
            groups = filtered_library[cat_key]

            if not groups:
                if search_lower:
                    st.caption(f"No documents in this category match \"{search}\".") 
                else:
                    st.caption("No documents in this category.")
                continue

            # ── Card grid: 2 columns ─────────────────────────────────────────
            col_a, col_b = st.columns(2, gap="medium")
            for gi, group in enumerate(groups):
                col = col_a if gi % 2 == 0 else col_b
                with col:
                    render_doc_card(group, card_key=f"{cat_key}_{gi}")





# ---------------------------------------------------------------------------
# AI Assistant
# ---------------------------------------------------------------------------

ILIAD_MAX_TOKENS = 8192  # upper bound on the model's reply length

SYSTEM_PROMPT = """You are a scientific data assistant for AbbVie's soft tissue and material testing team. You have access to two sources of information:

1. PRODUCT DATA: lift capacity measurements over time, material properties (HA concentration, elasticity, cohesivity, water uptake, extrusion force), and statistical comparisons between products. This data is provided under "## Product Performance Data" on every turn — use it freely to answer numerical questions.

2. LIBRARY DOCUMENTS: internal study protocols, technical reports, test methods, work instructions, and external publications (peer-reviewed literature used as supporting references). These will be provided to you with their exact document titles and categories in the format "### Document Title [Category]". When citing, note whether a source is an internal study/protocol or an external publication, since users treat these differently.

CRITICAL RULES FOR REFERENCES:
- Only cite documents that are explicitly provided to you in the context of this conversation.
- When asked for references, ONLY use the exact document titles shown under "### " headers in the provided context.
- NEVER invent, guess, or extrapolate document IDs, report numbers, or study numbers that were not explicitly given to you.
- If you are not sure which document a piece of information came from, say so rather than guessing.
- If asked for references and none were provided in context, say "I only have access to the documents retrieved for this session — please ask about a specific document by name."

Answer with maximum specificity based on the context. Include exact numerical values, timepoints, statistical findings, p-values, product concentrations, animal models, and explicit conclusions. Never give vague answers when specific data is available in the context. If specific numbers or findings appear in the documents, include them in your answer.

FORMAT: Prefer bullet points, tables, and short labelled sections over prose paragraphs — same information, faster to read. Skip preambles ("Great question", "Based on the documents provided"), skip restating the question, skip closing suggestions ("Let me know if you need more details"). Start with the answer."""


def _build_data_context(
    df: pd.DataFrame,
    timepoints: list[str],
    product_to_ref: dict[str, str],
    product_properties: pd.DataFrame,
) -> str:
    """
    Summarise the full dataset into a compact text block for LLM context.
    Includes all products' mean lift values and material properties.
    """
    lines = ["## Product Performance Data\n"]

    # Lift capacity means per product
    pivot = df.pivot_table(index="Product", columns="Timepoint", values="Average", aggfunc="mean", observed=True)
    pivot_std = df.pivot_table(index="Product", columns="Timepoint", values="StdDev", aggfunc="mean", observed=True)

    lines.append("### Lift capacity (mean ± SD) by timepoint\n")
    for product in pivot.index:
        ref = product_to_ref.get(product, "")
        label = f"{product} ({ref})" if ref else product
        row_parts = []
        for tp in timepoints:
            if tp in pivot.columns and pd.notna(pivot.loc[product, tp]):
                m = pivot.loc[product, tp]
                s = pivot_std.loc[product, tp] if tp in pivot_std.columns else float("nan")
                sd_str = f"±{s:.3f}" if pd.notna(s) else ""
                row_parts.append(f"{tp}: {m:.3f}{sd_str}")
        lines.append(f"- {label}: {', '.join(row_parts)}")

    # Material properties
    lines.append("\n### Material properties\n")
    for product in product_properties.index:
        props = []
        for col in STATIC_PROPERTIES:
            val = product_properties.loc[product, col]
            if pd.notna(val):
                try:
                    props.append(f"{col}={float(val):.3g}")
                except (ValueError, TypeError):
                    pass
        if props:
            lines.append(f"- {product}: {', '.join(props)}")

    return "\n".join(lines)


def _retrieve_library_context(question: str) -> list[dict]:
    """
    Keyword-only retrieval used as a fallback when the vector index is
    absent or returns no hits. Returns the same shape as VectorStore.search
    so the AI-Assistant pipeline can treat both sources uniformly.

    Scoring strategy:
      10 pts — exact study number/ID match in title (e.g. 1746-D52-070)
       5 pts — product name or key noun matches title exactly
       3 pts — any question keyword (>3 chars) matches title
       1 pt  — any question keyword (>3 chars) matches preview text

    When a specific study number is detected, return up to 3 documents.
    Otherwise up to 8.
    """
    import json
    index_path = Path(LIBRARY_PATH) / "library_index.json"
    if not index_path.exists():
        return []

    try:
        with open(index_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []

    entries = [e for e in data.get("entries", []) if not e.get("deleted")]
    if not entries:
        return []

    q_lower   = question.lower()
    study_nums = re.findall(
        r"[A-Za-z0-9]+-[A-Za-z0-9]+-[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*", question
    )
    # All meaningful words from the question (length > 3, not stopwords)
    stopwords = {"what", "were", "with", "from", "that", "this", "they",
                 "have", "about", "provide", "tell", "give", "show", "find",
                 "data", "study", "report", "document", "references", "reference",
                 "where", "which", "some", "their", "like", "also", "more"}
    q_words = {
        w for w in re.sub(r"[^a-z0-9 ]", "", q_lower).split()
        if len(w) > 3 and w not in stopwords
    }

    # Specific study number search → return fewer but more precise docs
    # Product/keyword search → return more docs for comprehensive coverage
    max_docs = 3 if study_nums else 8

    # Separate short "navigational" words (study numbers, acronyms)
    # from longer descriptive words (product names, procedures)
    # Product names are typically 4+ chars and not generic stopwords
    product_words = {w for w in q_words if len(w) >= 4}

    scored = []
    for entry in entries:
        title   = (entry.get("display_name") or entry.get("title", "")).lower()
        preview = entry.get("preview", "").lower()
        score   = 0

        # Exact study number in title — strongest signal
        for sn in study_nums:
            if sn.lower() in title:
                score += 10

        # Keyword matches in title — weighted by specificity
        for w in q_words:
            if w in title:
                score += max(3, len(w) - 1)

        # Product name in preview — score the same as a title match
        # This ensures docs where product is only in body text rank equally
        for w in product_words:
            if w in preview:
                # Count occurrences — more mentions = more relevant
                count = preview.count(w)
                score += min(count * 2, max(3, len(w) - 1))

        if score > 0:
            scored.append((score, entry))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Deduplicate revisions: keep only the highest-scoring entry per unique
    # base document. e.g. "AGN-2021-TR-048 Rev2" and "AGN-2021-TR-048 Rev1"
    # collapse to a single entry.
    seen_bases: set[str] = set()
    deduped = []
    for score, entry in scored:
        raw_title = entry.get("display_name") or entry.get("title", "")
        base = base_title(raw_title)
        if base not in seen_bases:
            seen_bases.add(base)
            deduped.append((score, entry))
        if len(deduped) >= max_docs:
            break

    if not deduped:
        return []

    results: list[dict] = []
    for score, entry in deduped:
        results.append({
            "title":    entry.get("display_name") or entry.get("title", ""),
            "category": entry.get("category", ""),
            "text":     entry.get("preview", ""),
            "score":    float(score),
            "source":   "keyword",
        })
    return results


# Per-document full-text cap when injecting exact-ID matches. Sized so a
# full protocol + report pair fits comfortably inside claude-4.5-sonnet's
# 200k token window, with room for conversation history and the reply.
STUDY_FULLTEXT_CHAR_CAP = 70000

# Total character budget for the assembled user message (library context +
# product data + question). claude-4.5-sonnet has a 200k token window;
# ~600k chars ≈ 150k tokens, leaving ~50k tokens for the system prompt,
# conversation history, and the model's 8k-token reply.
USER_MSG_CHAR_BUDGET = 600000


@st.cache_resource(show_spinner=False, max_entries=64)
def _cached_extract_text_impl(filepath: str, mtime: float) -> str:
    from doc_text import extract_text
    return extract_text(filepath)


def _cached_extract_text(filepath: str) -> str:
    """
    Module-level cache (bounded to 64 entries) for expensive PDF/DOCX parsing,
    keyed by (filepath, mtime). Avoids storing extracted text in session_state,
    which Streamlit pickles on every rerun and would grow unbounded.
    """
    try:
        mtime = Path(filepath).stat().st_mtime
    except OSError:
        return ""
    return _cached_extract_text_impl(filepath, mtime)


def _retrieve_study_fulltext(question: str) -> list[dict]:
    """
    For every study number in the question, read the complete extracted text
    of the matching library entries from disk and return them as chunk-shaped
    dicts. The AI assistant prepends these to whatever the vector search
    returns so direct ID lookups never miss a detail.
    """
    import json
    from concurrent.futures import ThreadPoolExecutor
    from doc_text import SUPPORTED_EXTS

    study_nums = re.findall(
        r"[A-Za-z0-9]+-[A-Za-z0-9]+-[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*", question
    )
    if not study_nums:
        return []

    index_path = Path(LIBRARY_PATH) / "library_index.json"
    if not index_path.exists():
        return []
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []

    entries = [e for e in data.get("entries", []) if not e.get("deleted")]
    needles = [s.lower() for s in study_nums]

    # Collect all filepaths we need to extract, across all matching entries
    matched_entries: list[tuple[dict, list[str]]] = []
    seen_ids: set[str] = set()
    all_filepaths: list[str] = []
    for entry in entries:
        title = (entry.get("display_name") or entry.get("title", "")).lower()
        if not any(n in title for n in needles):
            continue
        entry_id = entry.get("id", "")
        if entry_id in seen_ids:
            continue
        seen_ids.add(entry_id)

        fps = [
            f.get("filepath", "") for f in entry.get("files", [])
            if (f.get("ext") or "").lower() in SUPPORTED_EXTS and f.get("filepath")
        ]
        if fps:
            matched_entries.append((entry, fps))
            all_filepaths.extend(fps)

    if not all_filepaths:
        return []

    # Extract all files in parallel. The cache keeps hits cheap; misses run
    # on worker threads so multiple large PDFs don't serialize.
    extracted: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for fp, text in zip(all_filepaths, pool.map(_cached_extract_text, all_filepaths)):
            extracted[fp] = text

    matched: list[dict] = []
    for entry, fps in matched_entries:
        parts = [
            f"[{Path(fp).name}]\n{extracted[fp]}"
            for fp in fps if extracted.get(fp)
        ]
        combined = "\n\n".join(parts)
        if not combined:
            continue
        if len(combined) > STUDY_FULLTEXT_CHAR_CAP:
            combined = combined[:STUDY_FULLTEXT_CHAR_CAP] + "\n\n[... truncated ...]"

        matched.append({
            "title":    entry.get("display_name") or entry.get("title", ""),
            "category": entry.get("category", ""),
            "text":     combined,
            "score":    1.0,
            "source":   "fulltext",
        })

    return matched


def _extract_chunk_text(payload: dict) -> str:
    """
    Pull text out of a single streamed SSE payload. Handles Claude's native
    delta format and OpenAI-style delta format. Returns "" for non-text
    events (message_start, content_block_stop, usage updates, etc.).
    """
    # Claude: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "..."}}
    delta = payload.get("delta")
    if isinstance(delta, dict):
        if delta.get("type") == "text_delta" and "text" in delta:
            return delta["text"]
        # OpenAI-style: {"delta": {"content": "..."}}
        if isinstance(delta.get("content"), str):
            return delta["content"]
    # Some gateways flatten to {"text": "..."}
    if isinstance(payload.get("text"), str):
        return payload["text"]
    return ""


def _stream_iliad(
    messages: list[dict],
    max_tokens: int | None = None,
    timeout: int = 180,
):
    """
    Generator yielding text chunks from the ILIAD streaming chat endpoint.
    Falls back to a non-streaming request if SSE parsing fails. Callers can
    pass a larger max_tokens and timeout for long-running generation tasks
    (e.g. the Report Generator, where 16k+ output tokens take several
    minutes to stream).
    """
    if not iliad_client.get_api_key():
        yield "⚠️ ILIAD_API_KEY environment variable is not set. Please set it and restart the app."
        return

    effective_max = max_tokens or ILIAD_MAX_TOKENS

    try:
        resp = iliad_client.post_chat(
            messages, max_tokens=effective_max, stream=True, timeout=timeout
        )
        resp.raise_for_status()
        resp.encoding = "utf-8"

        emitted_any = False
        buffered_data: list[str] = []

        for raw in resp.iter_lines(decode_unicode=True):
            if raw is None:
                continue
            # Some servers ignore decode_unicode and still yield bytes
            if isinstance(raw, bytes):
                try:
                    raw = raw.decode("utf-8")
                except UnicodeDecodeError:
                    continue
            line = raw.strip()
            if not line:
                # Blank line terminates an SSE event — parse accumulated data
                if buffered_data:
                    payload_str = "".join(buffered_data).strip()
                    buffered_data = []
                    if payload_str == "[DONE]":
                        break
                    try:
                        payload = json.loads(payload_str)
                    except Exception:
                        continue
                    chunk = _extract_chunk_text(payload)
                    if chunk:
                        emitted_any = True
                        yield chunk
                continue
            if line.startswith(":"):
                continue  # SSE comment/heartbeat
            if line.startswith("data:"):
                buffered_data.append(line[5:].lstrip())
            elif line.startswith("event:"):
                continue  # event type handled by payload shape
            else:
                buffered_data.append(line)

        # Flush any trailing payload that wasn't terminated by a blank line
        if buffered_data:
            payload_str = "".join(buffered_data).strip()
            if payload_str and payload_str != "[DONE]":
                try:
                    payload = json.loads(payload_str)
                    chunk = _extract_chunk_text(payload)
                    if chunk:
                        emitted_any = True
                        yield chunk
                except Exception:
                    pass

        if emitted_any:
            return

        # Streaming produced no parseable text — fall through to non-streaming
        # so the user always gets an answer.
        for chunk in _call_iliad_nonstreaming(
            messages, max_tokens=effective_max, timeout=timeout,
        ):
            yield chunk

    except requests.exceptions.Timeout:
        yield "⚠️ Request timed out. The API may be busy — please try again."
    except requests.exceptions.HTTPError as e:
        yield f"⚠️ API error {e.response.status_code}: {e.response.text}"
    except Exception as e:
        yield f"⚠️ Streaming error: {e}"


def _call_iliad_nonstreaming(
    messages: list[dict],
    max_tokens: int | None = None,
    timeout: int = 180,
):
    """Single-shot request used as a streaming fallback. Yields one string."""
    try:
        resp = iliad_client.post_chat(
            messages,
            max_tokens=max_tokens or ILIAD_MAX_TOKENS,
            stream=False,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        if "content" in data and isinstance(data["content"], list):
            parts = [
                blk.get("text", "") for blk in data["content"]
                if isinstance(blk, dict) and blk.get("type") == "text"
            ]
            joined = "".join(parts).strip()
            if joined:
                yield joined
                return
        if "completion" in data and isinstance(data["completion"], dict):
            yield data["completion"].get("content", str(data)); return
        if "choices" in data:
            yield data["choices"][0]["message"]["content"]; return
        if "message" in data and isinstance(data["message"], dict):
            yield data["message"].get("content", str(data["message"])); return
        if "message" in data:
            yield data["message"]; return
        if "content" in data:
            yield data["content"]; return
        yield str(data)

    except requests.exceptions.Timeout:
        yield "⚠️ Request timed out. The API may be busy — please try again."
    except requests.exceptions.HTTPError as e:
        yield f"⚠️ API error {e.response.status_code}: {e.response.text}"
    except Exception as e:
        yield f"⚠️ Unexpected error: {e}"


def _call_iliad(messages: list[dict]) -> str:
    """Non-streaming convenience wrapper; returns the joined reply."""
    return "".join(_call_iliad_nonstreaming(messages))


# ---------------------------------------------------------------------------
# Report Generator
# ---------------------------------------------------------------------------

REPORT_PROMPTS = {
    "Study summary": """You are producing a CONCISE STUDY SUMMARY for AbbVie's soft-tissue and material testing team based on an uploaded document and relevant library context.

The library context contains TWO distinct source types, and you must treat them differently:
- **Internal studies** (categories: Study Reports, Study Protocols, Test Methods, Work Instructions) — the team's own prior experimental work. Compare head-to-head against current findings.
- **Publications** (category: Publications) — external peer-reviewed literature. Use these as supporting references to contextualize mechanisms, benchmark against published norms, and strengthen conclusions.

Output structure (markdown, use these exact H2 headings):

## Objectives
## Methods
## Key Findings
## Comparison to Prior Internal Studies
## Supporting Literature
## Conclusions

RULES:
- Keep total length under ~1000 words. Be dense and specific — numbers, timepoints, sample sizes, p-values, identifiers.
- Methods section: 3-6 bullet points summarising experimental setup (animals/model, groups, timepoints, key techniques).
- Key Findings: bullet points with exact numerical results. Lead with magnitude.
- **Comparison to Prior Internal Studies**: head-to-head comparisons against the team's own prior work. For each major finding:
  - **[Endpoint/finding]:** current result → prior internal result (from "Exact Library Title") → relationship (agrees / extends / contradicts / novel).
  Example: **Lift capacity at 24w:** current 0.42 ± 0.03 → 0.38 ± 0.05 in *AGN-2019-TR-044 Harmonyca pivotal* → agrees within SD, slight upward trend.
  If no internal comparator exists for a finding, say "no direct internal comparator" — don't skip silently.
- **Supporting Literature**: cite any Publications from the library that support or contextualize the findings (mechanism, clinical relevance, published benchmarks). Format as bullets:
  - *Exact publication title* — one sentence explaining what it supports or contradicts.
  If no relevant publications were retrieved, write "No publications retrieved in scope."
- Only cite library documents that actually appear in the provided context. Use the exact title shown.
- If something is not in the uploaded document, say so explicitly rather than inventing.
- Do NOT include preambles, restatements of the task, or closing suggestions.""",

    "Full study report": """You are producing a POLISHED STUDY REPORT for AbbVie's soft-tissue and material testing team based on an uploaded document and relevant library context.

The library context contains TWO distinct source types, and you must treat them differently:
- **Internal studies** (categories: Study Reports, Study Protocols, Test Methods, Work Instructions) — the team's own prior experimental work. Use these for head-to-head quantitative comparison.
- **Publications** (category: Publications) — external peer-reviewed literature. Use these for mechanistic context, published benchmarks, and to strengthen Background and Discussion.

Output structure (markdown, use these exact H2 headings in this order):

## Background
(Study rationale; relevant prior work. Distinguish internal precedents from published literature. Cite Publications for mechanistic/clinical context and internal studies for methodological precedent. State the gap being addressed.)

## Methods
(All procedural detail from the uploaded document: animal/tissue model, group sizes, dosing, timepoints, techniques, instruments, blinding/randomization if mentioned.)

## Results
(Every specific numerical finding, statistic, and observation. Use markdown tables where the source is tabular. Include p-values, effect sizes, and sample sizes where given.)

## Comparison to Prior Internal Studies
(Structure as a markdown table with these columns:
| Endpoint / Metric | Current study | Prior internal study | Relationship | Notes |
One row per major endpoint. "Prior internal study" must cite the exact library title and the specific prior value/range. "Relationship" is one of: agrees, extends, contradicts, or novel. If no direct internal comparator exists for an endpoint, include the row with "no direct comparator" so the gap is explicit.
Follow the table with 2-3 paragraphs of interpretive prose synthesising cross-study patterns — does this new study replicate earlier work, push beyond it, or disagree, and why might that be?)

## Supporting Literature
(Publications from the library that contextualize the findings. Organise by theme where helpful — e.g. "Mechanism of action", "Clinical benchmarks", "Methodological validation". For each citation: *Exact publication title* — one-to-two sentence explanation of how it supports, contradicts, or contextualizes the current findings. If no Publications were retrieved in scope, write "No relevant publications retrieved in scope." rather than fabricating any.)

## Discussion
(Mechanistic interpretation. Draw on BOTH the internal comparison table and the supporting literature. Discuss limitations noted in the source. Explain any methodological differences that could account for divergent results vs. prior internal studies or vs. published norms.)

## Conclusions
(Key takeaways as bullet points with specific numbers.)

## References
(Two-part list. Format each entry as a bullet.
**Internal sources:**
- **Exact title** [Category] — one-sentence relevance.
**Publications:**
- *Exact title* — one-sentence relevance.
Include every library document you cited in the body. If one subsection has no entries, omit it.)

RULES:
- Include every specific number, p-value, sample size, timepoint, animal count, and identifier from the uploaded document.
- Cite library documents by their exact title. Only cite documents that actually appear in the provided context.
- NEVER fabricate a publication title, DOI, or author list. If no publications are in context, say so.
- If information is not in the uploaded document or the library context, flag with "[not reported in source]" rather than inventing.
- Use markdown tables (pipe syntax) for tabular data in both Results and Comparison sections.
- Do NOT include preambles, restatements, or closing suggestions. Start with the first H2 heading.""",
}

# Reports can run long, especially table-heavy ones. Claude 4.5 Sonnet supports
# up to 64k output tokens; at ~60 tok/s streaming that would take ~18 minutes,
# which is well past any reasonable client timeout. 16k gives ample room for a
# full-report-with-tables while keeping generation under ~5 minutes.
REPORT_MAX_TOKENS = 16000

# How long to wait for the streaming report response before giving up. 10 min
# covers the worst-case "Claude spends a while processing 100k input tokens
# before first chunk arrives" scenario, plus full generation time.
REPORT_TIMEOUT_SEC = 600

# Library retrieval knobs for reports. Fewer docs + tighter per-doc cap keeps
# the input payload manageable so Claude starts streaming quickly.
REPORT_LIBRARY_TOPK = 8


def _markdown_to_docx(md: str, title: str) -> bytes:
    """
    Convert our model's markdown output into a .docx byte stream.
    Handles H1/H2/H3 headings, bullet/numbered lists, pipe tables, and
    **bold**/*italic* inline formatting. Enough for a polished report.
    """
    from io import BytesIO
    from docx import Document
    import re as _re

    doc = Document()
    doc.add_heading(title, level=0)

    def _add_runs(paragraph, text: str) -> None:
        """Parse **bold** and *italic* runs inside a paragraph."""
        pattern = _re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*)")
        pos = 0
        for m in pattern.finditer(text):
            if m.start() > pos:
                paragraph.add_run(text[pos:m.start()])
            token = m.group()
            if token.startswith("**"):
                run = paragraph.add_run(token[2:-2])
                run.bold = True
            else:
                run = paragraph.add_run(token[1:-1])
                run.italic = True
            pos = m.end()
        if pos < len(text):
            paragraph.add_run(text[pos:])

    lines = md.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        # Pipe table: look ahead for a separator line like |---|---|
        if (line.startswith("|") and i + 1 < len(lines)
                and _re.match(r"^\|[\s:\-|]+\|\s*$", lines[i + 1])):
            header_cells = [c.strip() for c in line.strip("|").split("|")]
            body_rows: list[list[str]] = []
            j = i + 2
            while j < len(lines) and lines[j].strip().startswith("|"):
                row = [c.strip() for c in lines[j].strip().strip("|").split("|")]
                body_rows.append(row)
                j += 1
            table = doc.add_table(rows=1 + len(body_rows), cols=len(header_cells))
            table.style = "Light Grid Accent 1"
            for c, h in enumerate(header_cells):
                cell = table.rows[0].cells[c]
                p = cell.paragraphs[0]
                run = p.add_run(h)
                run.bold = True
            for r, row in enumerate(body_rows, start=1):
                for c in range(len(header_cells)):
                    txt = row[c] if c < len(row) else ""
                    table.rows[r].cells[c].text = txt
            i = j
            continue

        if not line:
            i += 1
            continue

        if line.startswith("### "):
            doc.add_heading(line[4:].strip(), level=3)
        elif line.startswith("## "):
            doc.add_heading(line[3:].strip(), level=2)
        elif line.startswith("# "):
            doc.add_heading(line[2:].strip(), level=1)
        elif line.startswith("- ") or line.startswith("* "):
            p = doc.add_paragraph(style="List Bullet")
            _add_runs(p, line[2:].strip())
        elif _re.match(r"^\d+\.\s", line):
            p = doc.add_paragraph(style="List Number")
            _add_runs(p, _re.sub(r"^\d+\.\s", "", line))
        else:
            p = doc.add_paragraph()
            _add_runs(p, line)
        i += 1

    buf = BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.read()


# Cap on uploaded document text — protects the LLM context window.
REPORT_UPLOAD_CHAR_CAP = 200000


def _allocate_char_budget(sizes: list[int], total_budget: int) -> list[int]:
    """
    Split `total_budget` across files whose natural sizes are `sizes`,
    giving each file at least its equal share and letting small files
    donate their surplus back to larger ones. Guarantees every file gets
    at least 1 char of allocation (so empty allocations never silently
    drop a file).
    """
    n = len(sizes)
    if n == 0:
        return []
    if total_budget <= 0:
        return [0] * n

    # Sort by size ascending so small files claim their share first and
    # the remaining budget distributes over the larger ones.
    order = sorted(range(n), key=lambda i: sizes[i])
    allocations = [0] * n
    remaining = total_budget
    left = n

    for idx_pos, i in enumerate(order):
        if left <= 0:
            break
        share = remaining // left
        take = min(sizes[i], share)
        # Always leave at least 1 char so every file appears in context
        take = max(take, 1) if sizes[i] > 0 else 0
        allocations[i] = take
        remaining -= take
        left -= 1

    # Any leftover from small-file donations goes to the LARGEST file so
    # dense primary documents keep as much detail as possible.
    if remaining > 0:
        largest_i = max(range(n), key=lambda i: sizes[i])
        allocations[largest_i] += remaining
    return allocations


def render_report_generator() -> None:
    """
    Fourth tab — upload a PDF/DOCX/PPTX and generate a polished study
    summary or full study report contextualised by the library.
    """
    from doc_text import extract_text

    st.subheader("📝 Report Generator")
    st.caption(
        "Upload a study document (PDF, Word, or PowerPoint) and the assistant "
        "will synthesise a polished summary or full report, pulling context "
        "from your document library."
    )

    if not iliad_client.get_api_key():
        st.error("ILIAD_API_KEY not set — this tab needs the LLM to generate reports.")
        return

    vs: VectorStore | None = st.session_state.get("ai_vector_store")
    if vs is None or not vs.is_built():
        st.warning(
            "The vector index hasn't been built yet. Go to the **AI Assistant** "
            "tab and click **Build document index** first — the Report Generator "
            "uses the same index to find relevant prior studies."
        )

    # ── Upload ──────────────────────────────────────────────────────────────
    uploaded_files = st.file_uploader(
        "Upload one or more study documents",
        type=["pdf", "docx", "pptx"],
        accept_multiple_files=True,
        key="rg_uploader",
        help="PDF, Word (.docx), or PowerPoint (.pptx). Upload related "
             "documents together (e.g. protocol + report + amendment) and "
             "they'll be synthesised into one report.",
    )

    # Per-file extraction cache: {(name, size): extracted_text}. Adding a
    # new file never re-parses the files that are already cached.
    extracted_files: list[dict] = []
    if uploaded_files:
        file_cache = st.session_state.setdefault("_rg_file_cache", {})
        any_new = False
        for f in uploaded_files:
            key = (f.name, f.size)
            if key in file_cache:
                extracted_files.append({
                    "name": f.name, "size": f.size,
                    "text": file_cache[key],
                })
                continue
            any_new = True
            import tempfile, os
            suffix = Path(f.name).suffix.lower() or ".bin"
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(f.getbuffer())
                    tmp_path = tmp.name
                with st.spinner(f"Extracting text from {f.name}…"):
                    text_i = extract_text(tmp_path)
            finally:
                if tmp_path:
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
            file_cache[key] = text_i
            extracted_files.append({
                "name": f.name, "size": f.size, "text": text_i,
            })

        # Any time the upload set changes, drop the previously generated report
        current_set_key = tuple(sorted((f.name, f.size) for f in uploaded_files))
        if st.session_state.get("_rg_last_set_key") != current_set_key:
            st.session_state["_rg_last_set_key"] = current_set_key
            st.session_state.pop("_rg_generated", None)
            st.session_state.pop("_rg_sources", None)

        # Evict cached extractions for files no longer in the uploader
        current_keys = {(f.name, f.size) for f in uploaded_files}
        for stale in list(file_cache.keys()):
            if stale not in current_keys:
                file_cache.pop(stale, None)

    # Combine extracted texts with clear per-file separators and a FAIR-SHARE
    # budget so one huge file can't starve the others.
    if extracted_files:
        non_empty = [f for f in extracted_files if f["text"].strip()]
        empty_names = [f["name"] for f in extracted_files if not f["text"].strip()]

        if not non_empty:
            st.error(
                "Could not extract text from any of the uploaded files. If "
                "they're scanned PDFs, run OCR first."
            )
            return

        if empty_names:
            st.warning(
                "No text could be extracted from: "
                + ", ".join(f"`{n}`" for n in empty_names)
                + ". These will be ignored."
            )

        # Per-file budget: equal share initially, but files smaller than their
        # share donate the surplus back to larger files. Ensures every file
        # contributes and big files still get as much room as possible.
        allocations = _allocate_char_budget(
            [len(f["text"]) for f in non_empty],
            REPORT_UPLOAD_CHAR_CAP,
        )

        pieces: list[str] = []
        any_truncated = False
        for i, (f, cap_i) in enumerate(zip(non_empty, allocations), start=1):
            body = f["text"]
            truncated_here = len(body) > cap_i
            if truncated_here:
                body = body[:cap_i] + "\n\n[... file truncated to fit shared budget ...]"
                any_truncated = True
            header = (
                f"=== FILE {i} OF {len(non_empty)}: {f['name']} ==="
                if len(non_empty) > 1
                else f"=== {f['name']} ==="
            )
            pieces.append(f"{header}\n{body}")
            f["_chars_used"] = min(len(f["text"]), cap_i)
            f["_truncated"]  = truncated_here

        text = "\n\n".join(pieces)
        upload_label = (
            f"{len(non_empty)} files"
            if len(non_empty) > 1 else non_empty[0]["name"]
        )

        total_size = sum(f["size"] for f in non_empty)
        total_extracted = sum(len(f["text"]) for f in non_empty)
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Files", f"{len(non_empty)}")
        col_b.metric("Total size", f"{total_size / 1024:,.0f} KB")
        col_c.metric(
            "Sent to model",
            f"{len(text):,} / {total_extracted:,} chars",
            help="Combined length sent to the LLM vs total extracted. "
                 "Budget is split fairly across files."
        )

        with st.expander("Uploaded files", expanded=any_truncated):
            for f in non_empty:
                flag = "  ⚠ truncated" if f.get("_truncated") else ""
                st.caption(
                    f"📄 **{f['name']}** — {f['size'] / 1024:,.0f} KB, "
                    f"{len(f['text']):,} chars extracted, "
                    f"{f.get('_chars_used', 0):,} chars sent{flag}"
                )
    else:
        text = ""
        upload_label = ""

    # ── Configuration ───────────────────────────────────────────────────────
    st.markdown("")
    output_type = st.radio(
        "Output type",
        options=list(REPORT_PROMPTS.keys()),
        horizontal=True,
        key="rg_output_type",
    )

    generate_disabled = not text.strip()
    if st.button(
        "✨ Generate report",
        type="primary",
        disabled=generate_disabled,
        key="rg_generate_btn",
    ):
        _run_report_generation(text, upload_label, output_type, vs)
    if output_type == "Full study report":
        st.caption(
            "_Full reports with tables typically take 3–5 minutes to stream. "
            "Output appears progressively — don't reload if the first few "
            "seconds look quiet._"
        )

    # ── Render previously generated report ──────────────────────────────────
    generated = st.session_state.get("_rg_generated")
    if generated:
        st.divider()
        st.markdown("#### Generated report")

        # Constrain report body to a readable column width — running full-page
        # makes long reports hard to scan. Side margins stay empty.
        _, center_col, _ = st.columns([1, 4, 1])
        with center_col:
            with st.container(border=True):
                st.markdown(generated["markdown"])

            sources = generated.get("sources") or []
            if sources:
                _render_sources(sources)

            # Download button — build .docx on demand
            docx_bytes = _markdown_to_docx(
                generated["markdown"],
                title=f"{generated['output_type']} — {generated['source_filename']}",
            )
            today = datetime.now().strftime("%Y-%m-%d")
            source_label = Path(generated["source_filename"]).stem
            safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", source_label)[:80] or "report"
            dl_col, clear_col = st.columns(2)
            with dl_col:
                st.download_button(
                    "⬇ Download as Word (.docx)",
                    data=docx_bytes,
                    file_name=f"{safe_stem}_{generated['output_type'].lower().replace(' ', '_')}_{today}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True,
                )
            with clear_col:
                if st.button("Clear report", key="rg_clear", use_container_width=True):
                    st.session_state.pop("_rg_generated", None)
                    st.session_state.pop("_rg_sources", None)
                    st.rerun()


def _run_report_generation(
    upload_text: str,
    upload_filename: str,
    output_type: str,
    vs: "VectorStore | None",
) -> None:
    """Run retrieval + LLM synthesis and persist the result for rendering."""

    status = st.status("Preparing context…", expanded=False)
    with status:
        # Per-file budgets were applied at upload time; no extra cap needed here
        capped = upload_text

        # Seed library retrieval with the uploaded document itself. Using the
        # first 8k chars gives enough signal for the embedding model without
        # burning API budget on the whole file.
        library_chunks: list[dict] = []
        if vs is not None and vs.is_built():
            seed = upload_text[:8000]
            st.write(f"Searching library for related prior studies (top {REPORT_LIBRARY_TOPK})…")
            try:
                library_chunks = vs.search(seed, top_k=REPORT_LIBRARY_TOPK) or []
            except Exception as e:
                st.write(f"Library search failed: {e}")
                library_chunks = []
            st.write(f"Retrieved {len(library_chunks)} related document(s).")

        lib_ctx = ""
        if library_chunks:
            lib_ctx = "## Relevant prior studies from the library\n\n" + "\n\n".join(
                f"### {c['title']} [{c.get('category','')}]\n{c['text']}"
                for c in library_chunks
            )

        system = REPORT_PROMPTS[output_type]

        # When multiple files were uploaded, the combined text already carries
        # per-file "=== FILE N OF M: name ===" headers. Give the model an
        # explicit instruction to integrate EVERY uploaded file — otherwise
        # it tends to anchor on the largest one and ignore the rest.
        is_multi = "=== FILE 1 OF " in capped
        if is_multi:
            header = (
                f"Uploaded: **{upload_filename}** — MULTIPLE related documents "
                f"provided below. CRITICAL: you must read and integrate data "
                f"from EVERY file, not just the longest. Each file contributes "
                f"unique information. If a file provides data that supersedes, "
                f"amends, or complements another, reflect that explicitly. In "
                f"Methods, Results, and Comparison sections, cite which file "
                f"each piece of information came from (e.g. \"per *Amendment "
                f"v2.docx*, the primary endpoint changed to…\")."
            )
            body_wrapper = capped
        else:
            header = f"Uploaded document: **{upload_filename}**"
            body_wrapper = (
                f"=== BEGIN UPLOADED DOCUMENT ===\n{capped}\n=== END UPLOADED DOCUMENT ==="
            )
        user_parts = [f"{header}\n\n{body_wrapper}"]
        if lib_ctx:
            user_parts.append(
                "Use the following library documents for context. These are "
                "the ONLY library documents you may cite by name.\n\n" + lib_ctx
            )
        user_parts.append(
            f"Produce the {output_type}."
        )

        # Hard budget guard — trim library context if total gets huge
        total_chars = len(system) + sum(len(p) for p in user_parts)
        if total_chars > USER_MSG_CHAR_BUDGET and lib_ctx:
            over_by = total_chars - USER_MSG_CHAR_BUDGET
            trimmed = lib_ctx[: max(0, len(lib_ctx) - over_by)] + \
                "\n\n[... library context truncated to fit context window ...]"
            user_parts[1] = (
                "Use the following library documents for context. These are "
                "the ONLY library documents you may cite by name.\n\n" + trimmed
            )

        user_content = "\n\n".join(user_parts)
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_content},
        ]
        st.write(f"Sending **{len(user_content):,}** characters to the model…")
    status.update(label=f"Composing {output_type.lower()}…", state="running")

    # Stream the response so user sees output as it generates. Reports get
    # a much larger max_tokens budget than chat (so table-heavy Full reports
    # never cut off mid-sentence) and a longer network timeout to match
    # realistic generation time for 10-15k-token outputs.
    markdown = st.write_stream(
        _stream_iliad(
            messages,
            max_tokens=REPORT_MAX_TOKENS,
            timeout=REPORT_TIMEOUT_SEC,
        )
    )
    status.update(label="Done.", state="complete", expanded=False)

    st.session_state["_rg_generated"] = {
        "markdown":        markdown,
        "output_type":     output_type,
        "source_filename": upload_filename,
        "sources":         library_chunks,
    }
    st.rerun()


def render_ai_assistant(
    df: pd.DataFrame,
    timepoints: list[str],
    product_to_ref: dict[str, str],
    product_properties: pd.DataFrame,
) -> None:
    """Render the AI assistant chat tab."""

    st.subheader("🤖 AI Assistant")

    # ── Chat history ─────────────────────────────────────────────────────────
    # Initialise vector store (one per session)
    if "ai_vector_store" not in st.session_state:
        vs = VectorStore(LIBRARY_PATH, iliad_client.get_api_key())
        if vs._load():
            st.session_state["ai_vector_store"] = vs
        else:
            st.session_state["ai_vector_store"] = None

    # Resolve status after init so the file is detected on first render
    vs: VectorStore | None = st.session_state.get("ai_vector_store")
    vs_built = vs is not None and vs.is_built()


    st.caption(
        "Ask questions about product performance data or library documents. "
        "Context from the dataset and relevant documents is automatically included."
    )

    if not iliad_client.get_api_key():
        st.error(
            "**ILIAD_API_KEY not found.**  \n"
            "Set the environment variable before starting the app:  \n"
            "`set ILIAD_API_KEY=your-key-here` (Windows CMD)  \n"
            "`$env:ILIAD_API_KEY='your-key-here'` (PowerShell)"
        )

    # ── Vector store status + build controls ─────────────────────────────────
    with st.expander(
        "📚 Document index" + (" ✅" if vs_built else " ⚠️ Not built"),
        expanded=not vs_built,
    ):
        if vs_built:
            # Count unique documents and total chunks from the in-memory metadata
            doc_count = len({m.get("entry_id") for m in (vs._metadata or [])})
            chunk_count = len(vs._metadata or [])
            built_at = datetime.fromtimestamp(
                vs.vector_file.stat().st_mtime
            ).strftime("%Y-%m-%d %H:%M")
            st.success(
                f"Index ready — **{doc_count}** documents, "
                f"**{chunk_count:,}** chunks. Built **{built_at}**."
            )
            if vs and vs.needs_rebuild():
                st.warning("Library has been updated since the index was built — consider rebuilding.")
        else:
            st.info(
                "The document index has not been built yet. "
                "Click below to embed all library documents for semantic search. "
                "This takes a few minutes but only needs to be done once "
                "(or when new documents are added)."
            )

        col_update, col_full = st.columns(2)
        with col_update:
            update_label = "⚡ Update index (new files only)" if vs_built else "⚡ Build document index"
            if st.button(update_label, use_container_width=True, key="ai_update_vs",
                         disabled=not iliad_client.get_api_key()):
                new_vs = VectorStore(LIBRARY_PATH, iliad_client.get_api_key())
                if vs_built:
                    new_vs._load()
                progress = st.progress(0, text="Starting…")
                def _cb(done, total, msg):
                    pct = int(done / max(total, 1) * 100)
                    progress.progress(pct, text=msg)
                with st.spinner("Updating index…"):
                    result = new_vs.update(progress_callback=_cb) if vs_built else new_vs.build(progress_callback=_cb)
                progress.empty()
                if "error" in result:
                    st.error(result["error"])
                else:
                    if vs_built:
                        st.success(
                            f"Updated: {result.get('new_documents', 0)} new documents, "
                            f"{result.get('removed', 0)} removed."
                        )
                    else:
                        st.success(
                            f"Index built: {result['chunks']} chunks from "
                            f"{result['documents']} documents."
                            + (f" ({result['errors']} errors)" if result['errors'] else "")
                        )
                    st.session_state["ai_vector_store"] = new_vs
                    st.rerun()
        with col_full:
            if vs_built:
                if st.button("🔄 Full rebuild", use_container_width=True, key="ai_build_vs",
                             disabled=not iliad_client.get_api_key(),
                             help="Re-embed everything from scratch. Use if document content has changed."):
                    new_vs = VectorStore(LIBRARY_PATH, iliad_client.get_api_key())
                    progress = st.progress(0, text="Starting…")
                    def _cb2(done, total, msg):
                        pct = int(done / max(total, 1) * 100)
                        progress.progress(pct, text=msg)
                    with st.spinner("Rebuilding full index…"):
                        result = new_vs.build(progress_callback=_cb2)
                    progress.empty()
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.success(
                            f"Rebuilt: {result['chunks']} chunks from {result['documents']} documents."
                        )
                        st.session_state["ai_vector_store"] = new_vs
                        st.rerun()

    if "ai_messages" not in st.session_state:
        st.session_state["ai_messages"] = []

    # Display conversation history, with sources rendered under each
    # assistant turn so users can verify any past answer, not just the last.
    for msg in st.session_state["ai_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                _render_sources(msg["sources"])

    # ── Starter prompts (empty state) ────────────────────────────────────────
    if not st.session_state["ai_messages"]:
        with st.container(border=True):
            st.markdown("**Try asking:**")
            examples = [
                "Which product has the highest lift capacity at 52 weeks?",
                "Compare the material properties of Voluma and Harmonyca.",
                "Summarise study 1745-D76-054 including endpoints, methods, and conclusions.",
                "What do the reports say about hydration performance over time?",
            ]
            cols = st.columns(2)
            for i, ex in enumerate(examples):
                with cols[i % 2]:
                    if st.button(ex, key=f"ai_example_{i}", use_container_width=True):
                        st.session_state["ai_queued_question"] = ex
                        st.rerun()

    # ── Input ────────────────────────────────────────────────────────────────
    typed = st.chat_input("Ask about your data or documents…")
    queued = st.session_state.pop("ai_queued_question", None)
    question = typed or queued

    if question:
        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state["ai_messages"].append({"role": "user", "content": question})

        with st.chat_message("assistant"):
            status = st.status("Preparing context…", expanded=False)
            with status:
                # Build context
                # Cache data context — expensive pivot, only rebuild once per session
                if "ai_data_context" not in st.session_state:
                    st.session_state["ai_data_context"] = _build_data_context(
                        df, timepoints, product_to_ref, product_properties
                    )
                data_ctx = st.session_state["ai_data_context"]

                # ── Semantic document retrieval ───────────────────────────────
                vs_instance: VectorStore | None = st.session_state.get("ai_vector_store")

                # Detect follow-up questions referencing the previous document.
                # Bare "it"/"its" were intentionally dropped — they matched too
                # many false positives like "is it ok" / "what time is it".
                followup_signals = {
                    "same study", "same report", "same document", "that study",
                    "that report", "that document", "this study", "this report",
                    "the study", "the report", "the document",
                }
                is_followup = any(sig in question.lower() for sig in followup_signals)
                has_study_num = bool(
                    re.findall(r"[A-Za-z0-9]+-[A-Za-z0-9]+-[A-Za-z0-9]+", question)
                )
                is_short_conversational = (
                    len(question.split()) < 12
                    and not has_study_num
                    and "ai_last_doc_chunks" in st.session_state
                )

                if has_study_num:
                    st.write("Reading matching study documents from disk…")
                fulltext_chunks: list[dict] = (
                    _retrieve_study_fulltext(question) if has_study_num else []
                )

                if (is_followup or is_short_conversational) and not has_study_num and "ai_last_doc_chunks" in st.session_state:
                    st.write("Reusing last retrieved documents (follow-up question)…")
                    chunks = st.session_state["ai_last_doc_chunks"]
                elif vs_instance is not None and vs_instance.is_built():
                    if fulltext_chunks:
                        top_k = 5
                    elif has_study_num:
                        top_k = 12
                    else:
                        top_k = 25
                    st.write(f"Searching library index (top {top_k} matches)…")
                    chunks = vs_instance.search(question, top_k=top_k)
                    if fulltext_chunks:
                        ft_titles = {c["title"] for c in fulltext_chunks}
                        chunks = [c for c in chunks if c["title"] not in ft_titles]
                    if not chunks and not fulltext_chunks:
                        chunks = _retrieve_library_context(question)
                    st.session_state["ai_last_doc_chunks"] = fulltext_chunks + chunks
                else:
                    st.write("Vector index not built — falling back to keyword search…")
                    chunks = _retrieve_library_context(question)
                    st.session_state["ai_last_doc_chunks"] = fulltext_chunks + chunks

                sections: list[str] = []
                if fulltext_chunks:
                    sections.append(
                        "## Exact study match (full text)\n\n"
                        + "\n\n".join(
                            f"### {c['title']} [{c.get('category','')}]\n{c['text']}"
                            for c in fulltext_chunks
                        )
                    )
                if chunks:
                    sections.append(
                        "## Relevant Library Documents\n\n"
                        + "\n\n".join(
                            f"### {c['title']} [{c.get('category','')}]\n{c['text']}"
                            for c in chunks
                        )
                    )
                library_ctx = "\n\n".join(sections)

                # ── Build messages ────────────────────────────────────────────
                messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                # Short history replay keeps follow-ups contextual without
                # re-sending every prior turn's full library context.
                messages.extend(
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state["ai_messages"][:-1][-2:]
                )

                fixed_chars = len(data_ctx) + len(question) + 200
                lib_budget = max(0, USER_MSG_CHAR_BUDGET - fixed_chars)
                if len(library_ctx) > lib_budget:
                    library_ctx = (
                        library_ctx[:lib_budget]
                        + "\n\n[... library context truncated to fit context window ...]"
                    )

                parts: list[str] = []
                if data_ctx:
                    parts.append(data_ctx)
                if library_ctx:
                    parts.append(
                        "The following documents were retrieved from the library for "
                        "this question. These are the ONLY documents you may cite as "
                        "references. Do not cite any other document IDs or report "
                        "numbers not shown here.\n\n"
                        f"{library_ctx}"
                    )
                parts.append(f"Question: {question}")
                user_content = "\n\n".join(parts)
                messages.append({"role": "user", "content": user_content})
                st.write(
                    f"Sending **{len(user_content):,}** characters to the model…"
                )
            status.update(label="Composing answer…", state="running")

            # Stream the response so text appears as it's generated
            reply = st.write_stream(_stream_iliad(messages))
            status.update(label="Done.", state="complete", expanded=False)

            # Render sources right under the answer
            sources_for_msg = list(st.session_state.get("ai_last_doc_chunks") or [])
            if sources_for_msg:
                _render_sources(sources_for_msg)

        st.session_state["ai_messages"].append({
            "role":    "assistant",
            "content": reply,
            "sources": sources_for_msg,
        })

    # ── Controls ─────────────────────────────────────────────────────────────
    if st.session_state["ai_messages"]:
        if st.button("🗑 Clear conversation", key="ai_clear"):
            st.session_state["ai_messages"] = []
            st.session_state.pop("ai_last_doc_chunks", None)
            st.rerun()


def _render_sources(sources: list[dict]) -> None:
    """Compact citation list shown under each assistant response."""
    if not sources:
        return
    # Deduplicate by title so a study with multiple revisions shows once
    seen: set[str] = set()
    unique: list[dict] = []
    for s in sources:
        t = s.get("title", "")
        if t and t not in seen:
            seen.add(t)
            unique.append(s)
    if not unique:
        return
    labels = [f"**{s['title']}**" + (f" _({s['category']})_" if s.get("category") else "")
              for s in unique]
    st.caption("📄 Sources: " + " · ".join(labels))

# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="AbbVie Testing Dashboard",
        layout="wide",
        page_icon="🧪",
        menu_items={
            "About": (
                "**AbbVie Testing Dashboard**\n\n"
                "Compare product performance data, browse the document library, "
                "and ask the AI assistant questions grounded in both."
            ),
        },
    )

    st.markdown("""
        <style>
            .block-container { padding-top: 1.2rem !important; }
            header[data-testid="stHeader"] { display: none; }
            h1 { margin-bottom: 0.15rem !important; }
            [data-testid="stCaptionContainer"] { margin-top: -0.25rem; }
        </style>
    """, unsafe_allow_html=True)

    # --- Load data ---
    try:
        df, timepoints, product_to_ref, product_properties = load_data()
    except FileNotFoundError:
        st.error(
            f"Data file not found at the configured path:\n\n`{FILE_PATH}`\n\n"
            "Check that the network drive is mounted and the path is correct."
        )
        st.stop()
    except Exception as e:
        st.error(
            f"Could not load data from `{FILE_PATH}`:\n\n`{e}`\n\n"
            "Confirm the workbook matches the expected column layout "
            "(Product, 6 timepoints, 6 SDs, Reference, 5 property columns)."
        )
        st.stop()

    products = sorted(df["Product"].unique().tolist())

    st.title("AbbVie – Testing Dashboard")
    st.caption(
        f"**{len(products)}** products · **{len(timepoints)}** timepoints · "
        f"data source: `{Path(FILE_PATH).name}`"
    )

    tab_comparator, tab_library, tab_ai, tab_report = st.tabs([
        "📈 Product Comparator",
        "📚 Document Library",
        "🤖 AI Assistant",
        "📝 Report Generator",
    ])

    with tab_comparator:
        render_comparator(df, timepoints, product_to_ref, product_properties, products)

    with tab_library:
        render_library()

    with tab_ai:
        render_ai_assistant(df, timepoints, product_to_ref, product_properties)

    with tab_report:
        render_report_generator()


def render_comparator(
    df: pd.DataFrame,
    timepoints: list[str],
    product_to_ref: dict[str, str],
    product_properties: pd.DataFrame,
    products: list[str],
) -> None:
    """Render the Product Comparator tab body."""
    st.markdown(
        "Compare lift capacity over time and review material properties "
        "for selected products."
    )

    # ── Inline controls ──────────────────────────────────────────────────
    ctrl_col, n_col, reload_col = st.columns([4, 1.5, 1])
    with ctrl_col:
        default_selection = ["Voluma"] if "Voluma" in products else []
        selected_products = st.multiselect(
            "Products",
            options=products,
            default=default_selection,
            help="Select one or more products to compare.",
            label_visibility="visible",
        )
    with n_col:
        n_per_group = st.number_input(
            "n per group",
            min_value=2,
            max_value=500,
            value=10,
            step=1,
            help=(
                "Sample size used in t-test calculations. "
                "Set to the actual number of replicates per product per timepoint."
            ),
        )
    with reload_col:
        st.markdown("<div style='margin-top:1.75rem'></div>", unsafe_allow_html=True)
        if st.button("↻ Reload data", use_container_width=True,
                     help=f"Clear cache and reload from:\n{FILE_PATH}"):
            st.cache_data.clear()
            st.session_state.pop("ai_data_context", None)
            st.rerun()

    st.divider()

    if not selected_products:
        st.info("Select one or more products above to begin.")
        return

    filtered = df[df["Product"].isin(selected_products)].copy()
    pivot_avg, pivot_std = build_pivot_tables(filtered)
    valid_tps, skipped_tps = valid_tp_list(
        timepoints, pivot_avg, pivot_std, selected_products
    )

    # Selection summary — what's actually renderable from the current selection
    n_selected = len(selected_products)
    n_with_ts = sum(1 for p in selected_products if p in pivot_avg.index)
    n_with_props = sum(1 for p in selected_products if p in product_properties.index)
    st.caption(
        f"**{n_selected}** selected · "
        f"**{n_with_ts}** with time-series data · "
        f"**{n_with_props}** with material properties"
    )

    # Only warn about the default n=10 when stats will actually run
    if n_per_group == 10 and n_with_ts >= 2:
        st.caption(
            "⚠️ Using default **n=10** for statistical tests — update the "
            "control above to match your actual replicate count if different."
        )

    # Products with no time-series rows at all (e.g. no lift data in the sheet)
    missing_ts = [p for p in selected_products if p not in pivot_avg.index]
    if missing_ts:
        st.warning(
            f"The following selected product(s) have no time-series data and will "
            f"be excluded from the lift chart and statistics: "
            f"**{', '.join(missing_ts)}**. Their material properties will still "
            f"appear in the radar chart if available."
        )

    if skipped_tps:
        st.warning(
            f"The following timepoints were excluded from statistical tests "
            f"due to missing data: **{', '.join(skipped_tps)}**"
        )

    # --- Chart + properties ---
    col_chart, col_props = st.columns([2, 1.2], gap="large")

    with col_chart:
        fig = build_figure(
            filtered, pivot_avg, pivot_std, selected_products, product_to_ref
        )
        st.plotly_chart(fig, width="stretch")

        st.subheader("Time-series summary (mean ± SD)")
        if not pivot_avg.empty:
            present_in_pivot = [p for p in selected_products if p in pivot_avg.index and p in pivot_std.index]
            summary_display = pd.DataFrame(index=present_in_pivot)
            for col in pivot_avg.columns:
                summary_display[col] = [
                    format_mean_std(pivot_avg.loc[p, col], pivot_std.loc[p, col])
                    if pd.notna(pivot_avg.loc[p, col]) else "—"
                    for p in present_in_pivot
                ]
            st.dataframe(summary_display, width="stretch")

        # Export — single download button, bytes cached per selection
        export_key = (tuple(sorted(selected_products)), int(n_per_group))
        cache = st.session_state.get("_xlsx_export_cache") or {}
        if cache.get("key") != export_key:
            cache = {
                "key": export_key,
                "data": build_excel_export(
                    filtered, pivot_avg, pivot_std, product_properties,
                    selected_products, timepoints, product_to_ref,
                ),
            }
            st.session_state["_xlsx_export_cache"] = cache
        today = datetime.now().strftime("%Y-%m-%d")
        st.download_button(
            label="📥 Download Excel",
            data=cache["data"],
            file_name=f"product_comparison_{today}.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument"
                ".spreadsheetml.sheet"
            ),
        )
        # --- Statistics (inside left column to fill space beside properties) ---
        st.divider()
        st.subheader("Statistical comparisons")

        with st.expander("About the tests used"):
            st.markdown(
                """
                | Comparison | Test | When used |
                |---|---|---|
                | Two products, per timepoint | Independent-samples t-test | Always (2 products) |
                | 3+ products, per timepoint | One-way ANOVA (F-test) | When ≥ 3 products selected |
                | Two products, overall | Fisher's combined probability test | Always (≥ 2 products) |

                **Note:** t-tests use summary statistics (mean, SD, n) via
                `scipy.stats.ttest_ind_from_stats`. Set the correct *n* in the sidebar —
                the default of 10 may not match your experiment.
                The overall Fisher test combines per-timepoint p-values into one summary p-value per pair.
                p < 0.05 is used as the significance threshold throughout.
                """
            )

        if len(selected_products) >= 2 and valid_tps:
            present_for_stats = [p for p in selected_products
                                  if p in pivot_avg.index and p in pivot_std.index]

            if len(present_for_stats) < 2:
                st.info("At least two products with time-series data are required for pairwise comparisons.")
            else:
                # Per-timepoint pairwise t-tests
                st.markdown("#### Per-timepoint pairwise t-tests")
                tp_results = run_pairwise_ttests(
                    pivot_avg, pivot_std, present_for_stats, valid_tps, n_per_group
                )
                frames = [df.assign(Timepoint=tp) for tp, df in tp_results.items() if not df.empty]
                if frames:
                    all_tp = pd.concat(frames, ignore_index=True)
                    all_tp = all_tp[["Timepoint", "Pair", "p-value", "Significant?"]]
                    st.dataframe(all_tp, use_container_width=True, hide_index=True)
                    st.caption("_\"Significant?\" = p-value below the 0.05 threshold._")
                else:
                    st.info("No pairwise results to display.")

                # ANOVA (3+ products with data)
                if len(present_for_stats) >= 3:
                    st.markdown("#### One-way ANOVA (all selected products)")
                    anova_df = run_anova(filtered, present_for_stats, valid_tps)
                    st.dataframe(anova_df, use_container_width=True, hide_index=True)

                # Overall pairwise — Fisher's combined probability
                st.markdown("#### Overall pairwise significance (Fisher's combined test)")
                fisher_df = run_fisher_overall(
                    pivot_avg, pivot_std, present_for_stats, valid_tps, n_per_group
                )
                st.dataframe(fisher_df, use_container_width=True, hide_index=True)
                st.caption(
                    "Fisher's method combines the per-timepoint p-values into a single "
                    "test statistic. It answers: 'Is there a consistent difference between "
                    "these two products across all timepoints?' A significant result here "
                    "means the overall trend is unlikely to be due to chance, even if no "
                    "individual timepoint is significant on its own."
                )
        elif len(selected_products) < 2:
            st.info("Select at least two products to run statistical comparisons.")
        elif not valid_tps:
            st.warning("No valid timepoints found for statistical comparison.")

    with col_props:
        st.subheader("Material properties")
        valid_products_for_props = [
            p for p in selected_products if p in product_properties.index
        ]
        if valid_products_for_props:
            radar_fig = build_properties_radar(
                product_properties, selected_products, product_to_ref
            )
            if radar_fig:
                st.plotly_chart(radar_fig, width="stretch")

            st.caption(
                "Radar axes are normalised per attribute against all products in the "
                "dataset (not just selected), so axes stay stable as you add/remove "
                "products. Hover for true values."
            )

            bars_fig = build_properties_bars(
                product_properties, selected_products, product_to_ref
            )
            if bars_fig:
                st.plotly_chart(bars_fig, width="stretch")

            st.subheader("Product properties")
            props_display = product_properties.loc[valid_products_for_props].copy()
            for col in props_display.columns:
                props_display[col] = pd.to_numeric(
                    props_display[col], errors="coerce"
                ).round(3)
            st.dataframe(props_display.T, width="stretch")
        else:
            st.caption("No property data found for the selected products.")


if __name__ == "__main__":
    main()