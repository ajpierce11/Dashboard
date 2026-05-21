"""
auth.py
Who-is-this lookup and admin check for the dashboard.

Admins are the only users allowed to modify shared state on the network
drive (rescan the library, rebuild the vector index, organize files).
Non-admins see the same data but with write buttons hidden. This is not
a security boundary — anyone with code access can bypass it — it's a
UI guardrail that prevents multiple teammates stomping on each other's
writes to the shared `library_index.json` and `library_vectors.npz`.

CML reality check: CML's gateway stamps `Remote-User` only on the
deployer's own connections. Other authenticated viewers reach the app
with NO user-identifying header. So we use a two-track identity model:

  - Real identity (Remote-User) → only the deployer gets one. Used for
    admin gating against DASHBOARD_ADMINS.
  - Self-claimed identity → viewer picks their name from an admin-
    managed roster on first visit; choice persists in the
    `dashboard_user` cookie. Bookmarks live on the shared drive keyed
    on the typed name, so the same name brings the same profile across
    browsers and laptops. Trust-based; same trust model as the rest of
    the admin gating ("not a security boundary").

Configure admins by setting DASHBOARD_ADMINS in the environment (CML:
Project Settings → Environment Variables) to a comma-separated list of
AbbVie usernames. If the variable is unset the current user is treated
as admin in local dev only; on CML it fails closed.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path


_HEADER_CANDIDATES = (
    "Remote-User",
    "X-Remote-User",
    "X-Forwarded-User",
    "X-Forwarded-Preferred-Username",
    "X-Forwarded-Email",
    "Cdsw-User",
    "X-Cdsw-User",
    "X-Auth-Username",
    "X-User",
    "X-Knox-User",
    "Knox-User",
)

_ROSTER_NAME_RE = re.compile(r"[^A-Za-z0-9 ._-]")
USER_COOKIE = "dashboard_user"


def _streamlit_headers() -> dict | None:
    """Return st.context.headers as a normalized lowercase-key dict, or None."""
    try:
        import streamlit as st
        ctx = getattr(st, "context", None)
        headers = getattr(ctx, "headers", None) if ctx else None
        if not headers:
            return None
        out = {}
        for k, v in dict(headers).items():
            out[k.lower()] = v
        return out
    except Exception:
        return None


def _user_from_request_headers() -> str:
    """
    Real identity from gateway-stamped headers (Remote-User et al).

    On CML this only fires for the deployer's own connections; other
    viewers reach the app with no Remote-User header.
    """
    headers = _streamlit_headers()
    if not headers:
        return ""
    for name in _HEADER_CANDIDATES:
        v = headers.get(name.lower())
        if v:
            v = str(v).strip()
            if "@" in v:
                v = v.split("@", 1)[0]
            if v:
                return v
    return ""


def _parse_cookie_header(cookie: str) -> dict:
    out = {}
    for pair in cookie.split(";"):
        if "=" not in pair:
            continue
        k, v = pair.split("=", 1)
        out[k.strip()] = v.strip().strip('"')
    return out


def _user_from_cookie() -> str:
    """
    Identity the viewer chose for themselves on first visit.

    Set by `set_user_cookie()` when the viewer picks a name from the
    admin-managed roster. Persists ~1 year per browser; cleared on
    Switch-user. Sanitised on read so a forged cookie can't break
    file paths.
    """
    headers = _streamlit_headers()
    if not headers:
        return ""
    cookie = headers.get("cookie") or ""
    if not cookie:
        return ""
    parsed = _parse_cookie_header(cookie)
    name = parsed.get(USER_COOKIE, "").strip()
    if not name:
        return ""
    name = _ROSTER_NAME_RE.sub("", name).strip()[:60]
    return name


def current_user() -> str:
    """
    Return a stable per-viewer identifier suitable for keying private
    state (bookmarks, preferences). Order of resolution:

      1. Real identity from Remote-User (deployer only on CML).
      2. Roster name the viewer picked on first visit, persisted in
         the `dashboard_user` cookie. Profile follows the typed name
         across browsers/laptops since bookmarks live on the shared
         drive — clearing cookies just re-prompts.
      3. Local-dev env-var fallback (USERNAME / USER).
      4. Empty string → caller should show the login picker.

    Use `current_user_display()` for anything shown in the UI.
    Use `is_admin()` (which only honours #1) for permission checks.
    """
    real = _user_from_request_headers()
    if real:
        return real

    chosen = _user_from_cookie()
    if chosen:
        return chosen

    if not _on_cml():
        for var in ("USERNAME", "USER"):
            val = os.environ.get(var, "").strip()
            if val:
                return val

    return ""


def current_user_display() -> str:
    """
    Friendly name for UI ("Welcome, <X>"). Same as current_user(),
    but returns "" when the viewer hasn't identified themselves so
    the caller can show a generic greeting / route to the picker.
    """
    return current_user()


# ---------------------------------------------------------------------------
# Roster (admin-managed list of names viewers can pick from)
# ---------------------------------------------------------------------------

def _roster_path() -> Path:
    """Resolve roster.json under the configured Library/.user_prefs/."""
    # Lazy import so importing auth at module-load doesn't require config.
    from config import LIBRARY_PATH
    return Path(LIBRARY_PATH) / ".user_prefs" / "roster.json"


def load_roster() -> list[str]:
    """Return the configured roster, sorted and deduped. [] on any failure."""
    try:
        p = _roster_path()
        if not p.exists():
            return []
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            return []
        cleaned = sorted({
            _ROSTER_NAME_RE.sub("", str(x)).strip()
            for x in data if str(x).strip()
        })
        return [n for n in cleaned if n]
    except Exception:
        return []


def save_roster(names: list[str]) -> None:
    """Write a sanitised, sorted, deduped roster to disk. Admin-only caller."""
    p = _roster_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    cleaned = sorted({
        _ROSTER_NAME_RE.sub("", str(n)).strip()[:60]
        for n in names if str(n).strip()
    })
    cleaned = [n for n in cleaned if n]
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(cleaned, indent=2), encoding="utf-8")
    os.replace(tmp, p)


# ---------------------------------------------------------------------------
# Cookie set / clear (executed in the browser via a tiny JS component)
# ---------------------------------------------------------------------------

def _cookie_js(value: str | None) -> str:
    """
    Build the JS snippet that sets/clears the dashboard_user cookie
    and reloads the parent page so the next request carries it.
    Pass value=None to clear.
    """
    if value is None:
        body = (
            f"document.cookie = '{USER_COOKIE}=; Max-Age=0; Path=/; SameSite=Lax';"
        )
    else:
        safe = _ROSTER_NAME_RE.sub("", str(value)).strip()[:60]
        # Single-quote-safe (no apostrophes survive the regex above).
        body = (
            f"document.cookie = '{USER_COOKIE}={safe}; Max-Age=31536000; "
            f"Path=/; SameSite=Lax';"
        )
    return (
        "<script>"
        + body
        + "window.parent.location.reload();"
        "</script>"
    )


def set_user_cookie(name: str) -> None:
    """Set the dashboard_user cookie to `name` and reload. Streamlit-only."""
    try:
        import streamlit as st
        import streamlit.components.v1 as components
        components.html(_cookie_js(name), height=0)
    except Exception:
        pass


def clear_user_cookie() -> None:
    """Expire the dashboard_user cookie and reload."""
    try:
        import streamlit as st
        import streamlit.components.v1 as components
        components.html(_cookie_js(None), height=0)
    except Exception:
        pass


def _admin_list() -> list[str]:
    raw = os.environ.get("DASHBOARD_ADMINS", "")
    return [u.strip().lower() for u in raw.split(",") if u.strip()]


def _on_cml() -> bool:
    """True when the app is running inside Cloudera ML."""
    for var in ("CDSW_PROJECT", "CDSW_DOMAIN", "CDSW_APP_PORT", "CDSW_ENGINE_ID"):
        if os.environ.get(var):
            return True
    return False


def is_admin() -> bool:
    """
    True only when the viewer presents a real Remote-User identity
    (i.e. the deployer) AND it appears in DASHBOARD_ADMINS.

    Anon `viewer-...` IDs are never admin even if someone forges a
    matching cookie — the only way in is via the gateway-stamped
    Remote-User header, which CML controls.

    When DASHBOARD_ADMINS is unset:
      - Local dev (no CML env) → True, so single-user dev is frictionless.
      - CML → False. Forgetting to configure admins must fail closed.
    """
    admins = _admin_list()
    if not admins:
        return not _on_cml()
    real_user = _user_from_request_headers()
    return bool(real_user) and real_user.lower() in admins


def admin_contact() -> str:
    """
    Human-readable string naming the admin(s), for non-admin UI copy.
    Falls back to a generic message if nobody is configured.
    """
    admins = _admin_list()
    if not admins:
        return "an admin"
    if len(admins) == 1:
        return admins[0]
    return ", ".join(admins[:-1]) + f" or {admins[-1]}"
