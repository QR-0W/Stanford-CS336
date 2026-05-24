"""PII 脱敏：邮箱、电话号码、IPv4 地址的占位符替换。

对 Common Crawl 文本中的个人可识别信息（PII）用固定占位符替换，
避免模型在训练中记忆并复现真实联系方式。

每条函数返回 ``(替换后文本, 替换次数)``。
所有替换为不可逆操作（原信息丢失后无法恢复）。

替换占位符：
    - 邮箱   -> ``|||EMAIL_ADDRESS|||``
    - 电话   -> ``|||PHONE_NUMBER|||``
    - IPv4   -> ``|||IP_ADDRESS|||``
"""

from __future__ import annotations

import re


EMAIL_MASK = "|||EMAIL_ADDRESS|||"
PHONE_MASK = "|||PHONE_NUMBER|||"
IP_MASK = "|||IP_ADDRESS|||"


# 邮箱正则：标准 RFC 5321/5322 简化版，匹配 local@domain.tld 形式
# 使用负向 lookahead/lookbehind 避免匹配 URL path 中类似邮箱的片段
EMAIL_RE = re.compile(
    r"(?<![\w.!#$%&'*+/=?^_`{|}~-])"
    r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+"
    r"@"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+"
    r"(?![\w.!#$%&'*+/=?^_`{|}~-])"
)

# 电话正则：美国常见 10 位格式，支持括号和分隔符变体
# 例如 2831823829、(283)-182-3829、283-182-3829 等
PHONE_RE = re.compile(
    r"(?<!\d)"
    r"(?:\(?\d{3}\)?[-.\s]?)"
    r"\d{3}"
    r"[-.\s]?"
    r"\d{4}"
    r"(?!\d)"
)

# IPv4 正则：匹配 0.0.0.0 到 255.255.255.255
IPV4_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
IPV4_RE = re.compile(rf"(?<![\d.]){IPV4_OCTET}(?:\.{IPV4_OCTET}){{3}}(?!\.\d)(?!\d)")


def mask_emails(text: str) -> tuple[str, int]:
    """将文本中的邮箱地址替换为 ``|||EMAIL_ADDRESS|||``。"""
    return EMAIL_RE.subn(EMAIL_MASK, text)


def mask_phone_numbers(text: str) -> tuple[str, int]:
    """将文本中的美国格式电话号码替换为 ``|||PHONE_NUMBER|||``。"""
    return PHONE_RE.subn(PHONE_MASK, text)


def mask_ips(text: str) -> tuple[str, int]:
    """将文本中的 IPv4 地址替换为 ``|||IP_ADDRESS|||``。"""
    return IPV4_RE.subn(IP_MASK, text)
