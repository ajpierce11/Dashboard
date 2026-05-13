"""
insights.py
Analytical views over the AI-extracted metadata cache.

Nothing here calls the LLM — these are deterministic aggregations of
the structured fields ai_metadata.py produces (product, model,
endpoints, timepoints). They're cheap, reproducible, and easy to
spot-check.

Three primary outputs:
  - flatten(): one row per (study, product, model) triple so pandas
    aggregation is straightforward even when a study lists multiple
    products.
  - find_gaps(): combinations where coverage is conspicuously low
    relative to the rest of the matrix — "thing we'd expect to have
    tested but haven't."
  - endpoint_index(): endpoint -> list of studies that measured it,
    for the methodology navigator.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import pandas as pd


def flatten(meta_entries: dict) -> list[dict]:
    """
    Turn {entry_id: record} into a list of per-(study, product, model)
    rows. A study with two products becomes two rows; this keeps pandas
    groupby easy. Skips no-text and failed records.
    """
    rows: list[dict] = []
    for eid, rec in meta_entries.items():
        if not rec.get("ok", False):
            continue
        if rec.get("note") == "no_text":
            continue
        product_raw = rec.get("product", "") or ""
        products = [p.strip() for p in product_raw.split(",") if p.strip()]
        if not products:
            products = ["(unspecified)"]
        model = (rec.get("model", "") or "").strip() or "(unspecified)"
        for p in products:
            rows.append({
                "entry_id":   eid,
                "product":    p,
                "model":      model,
                "endpoints":  list(rec.get("endpoints", []) or []),
                "timepoints": list(rec.get("timepoints", []) or []),
            })
    return rows


def coverage_matrix(rows: list[dict]) -> pd.DataFrame:
    """Products (rows) × models (columns), cell = study count."""
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return (
        df.groupby(["product", "model"])["entry_id"]
        .nunique()
        .unstack(fill_value=0)
        .sort_index()
    )


def find_gaps(rows: list[dict], min_same_product_other_models: int = 2) -> list[dict]:
    """
    Surface (product, model) cells where a product has NEVER been tested
    in a given model but HAS been tested in at least `min_same_product_other_models`
    other models. These are the interesting gaps — the product is real
    and well-studied, just not in this model type.

    Returns a list of dicts sorted by how well-studied the product is
    overall (more-studied products → stronger signal their missing cell
    is a real gap worth investigating).
    """
    if not rows:
        return []

    matrix = coverage_matrix(rows)
    products = list(matrix.index)
    models = list(matrix.columns)

    gaps = []
    for product in products:
        if product.startswith("(unspecified"):
            continue
        models_covered = [m for m in models if matrix.loc[product, m] > 0]
        if len(models_covered) < min_same_product_other_models:
            continue
        total_studies = int(matrix.loc[product].sum())
        for model in models:
            if model.startswith("(unspecified"):
                continue
            if matrix.loc[product, model] == 0:
                gaps.append({
                    "product":         product,
                    "missing_model":   model,
                    "covered_models":  models_covered,
                    "total_studies":   total_studies,
                })

    gaps.sort(key=lambda g: (-g["total_studies"], g["product"], g["missing_model"]))
    return gaps


def endpoint_index(rows: list[dict]) -> dict[str, list[dict]]:
    """
    {endpoint (lowercased) → [{entry_id, product, model, timepoints}, ...]}

    Case-insensitive grouping so "Collagen I" / "collagen I" cluster.
    We intentionally don't do fuzzy/semantic normalization here — that
    would require a judgment call and is better surfaced in the UI
    (show the raw names so scientists can spot their own aliases).
    """
    idx: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        for ep in r["endpoints"]:
            key = ep.lower().strip()
            if not key:
                continue
            idx[key].append({
                "entry_id":   r["entry_id"],
                "product":    r["product"],
                "model":      r["model"],
                "timepoints": r["timepoints"],
                "display":    ep,  # preserve original casing for display
            })
    return dict(idx)


def endpoint_summary_table(rows: list[dict]) -> pd.DataFrame:
    """
    For the methodology navigator's top-level view — one row per unique
    endpoint with counts of distinct studies, products, and models that
    measured it.
    """
    idx = endpoint_index(rows)
    data = []
    for key, hits in sorted(idx.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        # Use the most common casing of the raw name for display
        display_names = [h["display"] for h in hits]
        best_display = max(set(display_names), key=display_names.count)
        data.append({
            "Endpoint":   best_display,
            "Studies":    len({h["entry_id"] for h in hits}),
            "Products":   len({h["product"] for h in hits}),
            "Models":     len({h["model"] for h in hits}),
        })
    return pd.DataFrame(data)


def studies_for_endpoint(rows: list[dict], endpoint_display: str) -> pd.DataFrame:
    """Detail view for one endpoint: all studies that measured it."""
    key = endpoint_display.lower().strip()
    idx = endpoint_index(rows)
    hits = idx.get(key, [])
    if not hits:
        return pd.DataFrame()
    return pd.DataFrame([
        {
            "Study":       h["entry_id"],
            "Product":     h["product"],
            "Model":       h["model"],
            "Timepoints":  ", ".join(h["timepoints"]) or "—",
        }
        for h in hits
    ]).drop_duplicates().reset_index(drop=True)


def coverage_summary_text(matrix: pd.DataFrame, gaps: list[dict], top_n: int = 8) -> str:
    """
    Render the coverage matrix + top gaps as a compact markdown block
    suitable for feeding to an LLM in a suggest-next-experiments prompt.
    Keeps token count down by truncating each dimension to top_n.
    """
    if matrix.empty:
        return "No coverage data available."

    # Trim to the most-studied products and models so the table stays legible.
    top_products = matrix.sum(axis=1).sort_values(ascending=False).head(top_n).index
    top_models = matrix.sum(axis=0).sort_values(ascending=False).head(top_n).index
    m = matrix.loc[top_products, top_models]

    lines = ["## Coverage matrix (study counts)\n"]
    lines.append("| Product | " + " | ".join(str(x) for x in m.columns) + " |")
    lines.append("|" + "---|" * (len(m.columns) + 1))
    for p in m.index:
        cells = [str(int(m.loc[p, mo])) for mo in m.columns]
        lines.append(f"| {p} | " + " | ".join(cells) + " |")

    lines.append("\n## Top gaps (well-studied products missing one model)\n")
    for g in gaps[:top_n]:
        lines.append(
            f"- **{g['product']}** has {g['total_studies']} studies "
            f"across {len(g['covered_models'])} model(s) but none in "
            f"**{g['missing_model']}**."
        )

    return "\n".join(lines)
