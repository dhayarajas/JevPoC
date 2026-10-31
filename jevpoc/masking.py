"""Mask PII / CPNI before any state leaves the gateway.

Identifiers become stable tokens (HMAC prefix), so the same number maps to the
same token inside one deployment but cannot be reversed by the vendor.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
from typing import Any

_SALT = os.environ.get("JEV_MASK_SALT", "local-poc-salt").encode()

PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("SECRET", re.compile(r"(?i)\b(?:bearer\s+[\w.-]{10,}|(?:sk|pk|ghp|gho|xox[abp]|AKIA)[-_A-Za-z0-9]{10,})")),
    ("SECRET", re.compile(r"(?i)\b(?:password|passwd|secret|api[_-]?key|token)\b\s*[:=]\s*[^\s,;'\"}]+")),
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")),
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    ("IMEI", re.compile(r"\b\d{15}\b")),
    ("PHONE", re.compile(r"(?<!\w)(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}\b")),
    ("ACCT", re.compile(r"\b\d{9,12}(?:-\d{5})?\b")),
    ("SSN4", re.compile(r"(?i)\b(?:ssn|social)[^\d]{0,12}\d{4}\b")),
]

SECRET_KEY = re.compile(
    r"(?i)^(?:[\w-]*[_-])?(?:password|passwd|pwd|secret|api[_-]?key|apikey|token|auth|authorization"
    r"|credentials?|private[_-]?key|client[_-]?secret)$"
)
_MASKED_SECRET = re.compile(r"^<SECRET_[0-9a-f]{6}>$")

DROP_KEYS = {"msisdn", "imsi", "imei", "account_id", "customer_name", "email", "ssn_last4", "pin", "address"}


def token(kind: str, value: str) -> str:
    h = hmac.new(_SALT, value.encode(), hashlib.sha256).hexdigest()[:6]
    return f"<{kind}_{h}>"


def mask_text(text: str, names: list[str] | None = None) -> str:
    for kind, pat in PATTERNS:
        text = pat.sub(lambda m, k=kind: token(k, m.group(0)), text)
    for n in names or []:
        if n:
            text = re.sub(rf"\b{re.escape(n)}\b", "<NAME>", text, flags=re.IGNORECASE)
    return text


def mask(obj: Any, names: list[str] | None = None) -> Any:
    """Recursively drop identifier keys and mask free text."""
    if isinstance(obj, dict):
        return {
            k: token("SECRET", str(v)) if SECRET_KEY.match(k) and isinstance(v, (str, int)) else mask(v, names)
            for k, v in obj.items()
            if k.lower() not in DROP_KEYS
        }
    if isinstance(obj, list):
        return [mask(v, names) for v in obj]
    if isinstance(obj, str):
        return mask_text(obj, names)
    return obj


def _unmasked_secret_value(obj: Any) -> bool:
    if isinstance(obj, dict):
        return any(
            (SECRET_KEY.match(str(k)) and isinstance(v, (str, int)) and not _MASKED_SECRET.match(str(v)))
            or _unmasked_secret_value(v)
            for k, v in obj.items()
        )
    if isinstance(obj, list):
        return any(_unmasked_secret_value(v) for v in obj)
    return False


def contains_pii(obj: Any) -> bool:
    if _unmasked_secret_value(obj):
        return True
    s = str(obj)
    return any(p.search(s) for k, p in PATTERNS if k != "SSN4")
