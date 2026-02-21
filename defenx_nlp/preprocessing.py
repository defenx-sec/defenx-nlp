"""
defenx_nlp.preprocessing — Text cleaning and normalisation utilities.

All functions are pure (no side effects, no model loading) and are safe
to call from multiple threads simultaneously.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Optional


# ── Single-text operations ────────────────────────────────────────────────────

def normalize_whitespace(text: str) -> str:
    """
    Collapse all runs of whitespace (spaces, tabs, newlines) to a single space
    and strip leading/trailing whitespace.

    Examples
    --------
    >>> normalize_whitespace("  hello\\t\\nworld  ")
    'hello world'
    """
    return re.sub(r"\s+", " ", text).strip()


def normalize_unicode(text: str, form: str = "NFC") -> str:
    """
    Apply Unicode normalisation.

    Parameters
    ----------
    text : str
    form : str
        One of ``"NFC"``, ``"NFD"``, ``"NFKC"``, ``"NFKD"`` (default ``"NFC"``).

    Examples
    --------
    >>> normalize_unicode("café")
    'café'
    """
    return unicodedata.normalize(form, text)


def remove_urls(text: str) -> str:
    """
    Strip HTTP/HTTPS/FTP URLs from text.

    Examples
    --------
    >>> remove_urls("Visit https://example.com for details")
    'Visit  for details'
    """
    return re.sub(r"https?://\S+|ftp://\S+|www\.\S+", "", text)


def remove_emails(text: str) -> str:
    """Remove e-mail addresses from text."""
    return re.sub(r"\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b", "", text)


def remove_special_chars(
    text: str,
    keep_punct: bool = True,
    keep_digits: bool = True,
) -> str:
    """
    Remove non-alphanumeric characters.

    Parameters
    ----------
    text        : str
    keep_punct  : bool — retain ``.!?,;:'"-`` when True (default True)
    keep_digits : bool — retain 0–9 when True (default True)
    """
    if keep_punct and keep_digits:
        return re.sub(r"[^\w\s.!?,;:'\"()-]", "", text)
    if keep_punct:
        return re.sub(r"[^\D\s.!?,;:'\"()-]", "", text)
    if keep_digits:
        return re.sub(r"[^\w\s]", "", text)
    return re.sub(r"[^a-zA-Z\s]", "", text)


def truncate(text: str, max_chars: int = 512, ellipsis: bool = True) -> str:
    """
    Hard-truncate ``text`` to at most ``max_chars`` characters.

    Parameters
    ----------
    text      : str
    max_chars : int  (default 512 — matches all-MiniLM token budget)
    ellipsis  : bool — append ``"…"`` when text is cut (default True)

    Examples
    --------
    >>> truncate("hello world", max_chars=5)
    'hello…'
    """
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rstrip()
    return cut + "…" if ellipsis else cut


def clean_text(
    text: str,
    *,
    lowercase: bool = False,
    remove_urls_flag: bool = False,
    remove_emails_flag: bool = False,
    remove_special: bool = False,
    max_chars: Optional[int] = None,
) -> str:
    """
    Configurable text cleaning pipeline.

    Parameters
    ----------
    text               : str  — raw input
    lowercase          : bool — convert to lower case
    remove_urls_flag   : bool — strip URLs
    remove_emails_flag : bool — strip e-mail addresses
    remove_special     : bool — strip non-alphanumeric characters
    max_chars          : int | None — truncate if set

    Returns
    -------
    str — cleaned text

    Examples
    --------
    >>> clean_text("  HELLO  WORLD  ", lowercase=True)
    'hello world'
    >>> clean_text("Email me at x@y.com", remove_emails_flag=True)
    'Email me at '
    """
    text = normalize_unicode(text)
    text = normalize_whitespace(text)
    if lowercase:
        text = text.lower()
    if remove_urls_flag:
        text = remove_urls(text)
    if remove_emails_flag:
        text = remove_emails(text)
    if remove_special:
        text = remove_special_chars(text)
    if max_chars is not None:
        text = truncate(text, max_chars)
    return text


# ── Batch operations ──────────────────────────────────────────────────────────

def batch_clean(
    texts: List[str],
    **clean_kwargs,
) -> List[str]:
    """
    Apply :func:`clean_text` to every element of ``texts``.

    All keyword arguments are forwarded to :func:`clean_text`.

    Parameters
    ----------
    texts        : list of str
    **clean_kwargs : forwarded to ``clean_text``

    Returns
    -------
    list of str

    Examples
    --------
    >>> batch_clean(["Hello!", "  World  "], lowercase=True)
    ['hello!', 'world']
    """
    return [clean_text(t, **clean_kwargs) for t in texts]


def deduplicate(texts: List[str], case_sensitive: bool = True) -> List[str]:
    """
    Remove exact duplicate strings while preserving order.

    Parameters
    ----------
    texts          : list of str
    case_sensitive : bool (default True)

    Returns
    -------
    list of str — deduplicated, order preserved
    """
    seen = set()
    result = []
    for t in texts:
        key = t if case_sensitive else t.lower()
        if key not in seen:
            seen.add(key)
            result.append(t)
    return result


def filter_empty(texts: List[str], min_chars: int = 1) -> List[str]:
    """
    Remove strings shorter than ``min_chars`` (after stripping).

    Parameters
    ----------
    texts     : list of str
    min_chars : int (default 1)
    """
    return [t for t in texts if len(t.strip()) >= min_chars]
