"""
auth.py
Who-is-this lookup and admin check for the dashboard.

Admins are the only users allowed to modify shared state on the network
drive (rescan the library, rebuild the vector index, organize files).
Non-admins see the same data but with write buttons hidden. This is not
a security boundary — anyone with code access can bypass it — it's a
UI guardrail that prevents multiple teammates stomping on each other's
writes to the shared `library_index.json` and `library_vectors.npz`.

Configure admins by setting DASHBOARD_ADMINS in the environment (CML:
Project Settings → Environment Variables) to a comma-separated list of
AbbVie usernames. If the variable is unset the current user is treated
as admin so local development still works.
"""

from __future__ import annotations

import os


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


def _user_from_request_headers() -> str:
    """
    Pull the viewer's identity from the incoming HTTP request headers.

    On CML the app process is owned by whoever deployed it, so env vars
    like CDSW_USER reflect the deployer — not the person whose browser
    is currently hitting the app. CML's Knox gateway stamps the actual
    viewer into a request header; we try the names commonly used.
    Returns "" if nothing is found or Streamlit context isn't available.
    """
    try:
        import streamlit as st
        headers = getattr(st, "context", None)
        headers = getattr(headers, "headers", None) if headers else None
        if not headers:
            return ""
        # st.context.headers is dict-like and case-insensitive in practice;
        # normalize anyway so we don't miss a different casing.
        normalized = {}
        try:
            for k, v in dict(headers).items():
                normalized[k.lower()] = v
        except Exception:
            return ""
        for name in _HEADER_CANDIDATES:
            v = normalized.get(name.lower())
            if v:
                v = str(v).strip()
                # Some gateways pass an email — strip the @domain so it
                # matches the bare-username form used in DASHBOARD_ADMINS.
                if "@" in v:
                    v = v.split("@", 1)[0]
                if v:
                    return v
        return ""
    except Exception:
        return ""


def current_user() -> str:
    """
    Return the username of the person currently viewing the app.

    Order of resolution:
      1. HTTP request headers (CML / reverse-proxy injected) — this is
         the only source that distinguishes one viewer from another on
         a shared deployment.
      2. Local-dev env-var fallback (USERNAME / USER) — only used when
         not on CML, since on CML those vars resolve to the deployer.

    Returns an empty string if nothing is available; callers should
    treat that as "unknown viewer" (not admin, no bookmarks, etc.).
    """
    user = _user_from_request_headers()
    if user:
        return user

    if _on_cml():
        # On CML, refusing to fall back to env vars is the whole point:
        # CDSW_USER is the deployer, not the viewer, so using it would
        # make every teammate appear to be the deployer.
        return ""

    for var in ("USERNAME", "USER"):
        val = os.environ.get(var, "").strip()
        if val:
            return val
    return ""


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
    True when the current user is in DASHBOARD_ADMINS.

    Behaviour when DASHBOARD_ADMINS is unset:
      - Local dev (no CML env) → True, so single-user development is
        frictionless.
      - CML (any CDSW_* var set) → False. On a hosted multi-tenant
        deployment, forgetting to configure admins must fail closed,
        not open: otherwise every teammate gets write access to the
        shared library and the entire point of admin gating is lost.
    """
    admins = _admin_list()
    if not admins:
        return not _on_cml()
    return current_user().lower() in admins


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
