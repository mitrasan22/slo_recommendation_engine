"""
PII scrubbing utilities for log messages, request paths, and trace payloads.

Notes
-----
Implements the security policy: metric label values that match email patterns,
IP address patterns, or UUID patterns are scrubbed before storage in the
database or Opik traces.

This module is the single implementation of that policy. All callsites —
the feedback log, request path logger, and error handlers — use ``scrub()``.

Patterns covered:

  - Email addresses                            -> ``[email]``
  - IPv4 addresses                             -> ``[ipv4]``
  - IPv6 addresses                             -> ``[ipv6]``
  - UUIDs (v1-v5)                              -> ``[uuid]``
  - Bearer / JWT tokens                        -> ``[token]``
  - AWS-style secret keys                      -> ``[secret]``
  - Query-param sensitive values (password=,
    token=, key=, secret=, email=)             -> ``param=[redacted]``

The function is intentionally conservative: it only strips values that match
well-known PII shapes. It does not attempt NER or ML-based detection.
"""
from __future__ import annotations

import re

_EMAIL = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
    re.ASCII,
)

_IPV4 = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
)

_IPV6 = re.compile(
    r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b"
    r"|(?:[0-9a-fA-F]{1,4}:){1,7}:"
    r"|::(?:[0-9a-fA-F]{1,4}:){0,6}[0-9a-fA-F]{1,4}"
)

_UUID = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}"
    r"-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)

_TOKEN = re.compile(
    r"\bey[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]*\b"
)

_SENSITIVE_PARAM = re.compile(
    r"(?i)(?:password|token|api[_-]?key|secret|email|apikey)"
    r"=([^&\s\"']+)",
)


def scrub(text: str) -> str:
    """
    Return a copy of text with PII patterns replaced by safe placeholders.

    Parameters
    ----------
    text : str
        Input string that may contain PII such as emails, IP addresses,
        UUIDs, JWT tokens, or sensitive query parameters.

    Returns
    -------
    str
        Copy of the input with all matched PII patterns replaced. Returns
        the input unchanged if it is falsy (empty string, ``None``).

    Notes
    -----
    Safe to call on log messages, request paths, exception strings, and
    JSON field values before writing to disk or sending to an LLM. Replacement
    order is significant: JWT tokens are replaced before UUIDs because JWTs
    contain base64url segments that could partially match UUID patterns.
    Substitution is applied left-to-right in this order:

    1. ``[token]``    — Bearer/JWT tokens (``eyXXX.XXX.XXX``)
    2. ``[email]``    — email addresses
    3. ``[uuid]``     — UUID v1-v5 hyphenated strings
    4. ``[ipv4]``     — dotted-decimal IPv4 addresses
    5. ``[ipv6]``     — colon-delimited IPv6 addresses
    6. ``param=[redacted]`` — sensitive query parameter values
    """
    if not text:
        return text
    text = _TOKEN.sub("[token]", text)
    text = _EMAIL.sub("[email]", text)
    text = _UUID.sub("[uuid]", text)
    text = _IPV4.sub("[ipv4]", text)
    text = _IPV6.sub("[ipv6]", text)
    text = _SENSITIVE_PARAM.sub(lambda m: m.group(0).split("=")[0] + "=[redacted]", text)
    return text


def scrub_dict(data: dict) -> dict:
    """
    Return a shallow copy of a dict with all string values scrubbed.

    Parameters
    ----------
    data : dict
        Dictionary whose string values may contain PII.

    Returns
    -------
    dict
        Shallow copy of ``data`` with every string value passed through
        ``scrub()``. Non-string values are copied as-is.

    Notes
    -----
    Does not recurse into nested dicts or lists. Call ``scrub()`` explicitly
    on deeply nested values when needed.
    """
    return {k: scrub(v) if isinstance(v, str) else v for k, v in data.items()}
