from __future__ import annotations

import re


EMAIL_MASK = "|||EMAIL_ADDRESS|||"
PHONE_MASK = "|||PHONE_NUMBER|||"
IP_MASK = "|||IP_ADDRESS|||"


EMAIL_RE = re.compile(
    r"(?<![\w.!#$%&'*+/=?^_`{|}~-])"
    r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+"
    r"@"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+"
    r"(?![\w.!#$%&'*+/=?^_`{|}~-])"
)

PHONE_RE = re.compile(
    r"(?<!\d)"
    r"(?:\(?\d{3}\)?[-.\s]?)"
    r"\d{3}"
    r"[-.\s]?"
    r"\d{4}"
    r"(?!\d)"
)

IPV4_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
IPV4_RE = re.compile(rf"(?<![\d.]){IPV4_OCTET}(?:\.{IPV4_OCTET}){{3}}(?!\.\d)(?!\d)")


def mask_emails(text: str) -> tuple[str, int]:
    return EMAIL_RE.subn(EMAIL_MASK, text)


def mask_phone_numbers(text: str) -> tuple[str, int]:
    return PHONE_RE.subn(PHONE_MASK, text)


def mask_ips(text: str) -> tuple[str, int]:
    return IPV4_RE.subn(IP_MASK, text)
