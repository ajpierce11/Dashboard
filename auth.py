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


def current_user() -> str:
    """
    Return the username of the person running this Streamlit session.

    CML injects `HADOOP_USER_NAME` and/or `CDSW_USER`; locally we fall
    back to the standard shell env vars. Returns an empty string if
    nothing is set (shouldn't happen in practice).
    """
    for var in ("CDSW_USER", "HADOOP_USER_NAME", "USERNAME", "USER"):
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
