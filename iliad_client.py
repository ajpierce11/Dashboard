"""
iliad_client.py
Single place for talking to AbbVie's internal ILIAD LLM gateway.

Before this module existed, verify=False, urllib3 warning suppression, the
x-api-key header, and retry logic were repeated across vector_store.py and
comparator.py. Consolidating here means one place to change TLS handling,
auth, or retry policy, and it makes the API callers trivial to mock for
tests.

The gateway uses a self-signed internal cert, so verify=False is intentional
and scoped only to traffic for this host.
"""

from __future__ import annotations

import os
import time

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_URL = "https://iliad-emerging-api.abbvienet.com/api/v1"
CHAT_URL = f"{BASE_URL}/chat/claude-4.5-sonnet"
EMBED_URL = f"{BASE_URL}/embed/text-embedding-3-small"


def get_api_key() -> str:
    """Read the ILIAD API key from the environment."""
    return os.environ.get("ILIAD_API_KEY", "")


def _session() -> requests.Session:
    """A shared session keeps the HTTPS connection alive across calls."""
    s = requests.Session()
    s.verify = False
    return s


_SESSION: requests.Session | None = None


def session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        _SESSION = _session()
    return _SESSION


def _headers(api_key: str | None = None, stream: bool = False) -> dict[str, str]:
    h = {"x-api-key": api_key or get_api_key()}
    if stream:
        h["Accept"] = "text/event-stream"
    return h


def post_chat(
    messages: list[dict],
    *,
    max_tokens: int,
    stream: bool = False,
    timeout: int = 180,
) -> requests.Response:
    """
    POST to the ILIAD chat endpoint. Returns the raw Response; callers handle
    the response shape (SSE streaming vs. JSON) based on the `stream` flag.
    """
    payload: dict = {"messages": messages, "max_tokens": max_tokens}
    if stream:
        payload["stream"] = True
    return session().post(
        CHAT_URL,
        json=payload,
        headers=_headers(stream=stream),
        timeout=timeout,
        stream=stream,
    )


def embed(texts: list[str], api_key: str | None = None, timeout: int = 30) -> list[list[float]]:
    """
    Embed a batch of texts. Retries transient failures with exponential backoff.
    Handles both ILIAD's native {"embeddings": [...]} shape and the OpenAI
    compatibility shape {"data": [{"embedding": [...]}]}.
    """
    if not texts:
        return []

    key = api_key or get_api_key()
    last_exc: Exception | None = None
    for attempt in range(4):
        try:
            resp = session().post(
                EMBED_URL,
                json={"input": texts},
                headers={"x-api-key": key},
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            if "embeddings" in data:
                return data["embeddings"]
            if "data" in data:
                return [item["embedding"] for item in data["data"]]
            if isinstance(data, list):
                return data
            raise ValueError(f"Unexpected response shape: {list(data.keys())}")
        except Exception as e:
            last_exc = e
            if attempt < 3:
                time.sleep(2 ** attempt)
    if last_exc is not None:
        raise last_exc
    return []
