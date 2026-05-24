"""HTML 文本抽取：从 Common Crawl WARC 原始字节中提取纯文本。

使用 Resiliparse 库的 ``extract_plain_text``，并依次尝试多种字符编码
解码 HTML，避免因编码声明错误或缺失导致 UnicodeDecodeError。

编码回退顺序：UTF-8 -> Latin-1 -> CP1252 -> ISO-8859-1 -> UTF-8(replace)
"""

from resiliparse.extract.html2text import extract_plain_text


def extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    """从原始 HTML 字节中抽取纯文本。

    采用多编码回退策略：先依次尝试 UTF-8、Latin-1、CP1252、ISO-8859-1，
    若全部失败则用 UTF-8 + ``errors="replace"`` 强制解码（以保证 Reader 可继续）。
    最后调用 Resiliparse 的 ``extract_plain_text`` 剥离标签和格式。

    Args:
        html_bytes: WARC response body 的原始字节。

    Returns:
        解码并抽取后的纯文本；若结果为空字符串则返回 ``None``。
    """
    # 多编码回退：Common Crawl 中很多页面不是 UTF-8，尝试常见编码避免乱码
    for encoding in ("utf-8", "latin-1", "cp1252", "iso-8859-1"):
        try:
            html_str = html_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        # 全部解码失败，用 UTF-8 替换模式兜底，保证总能继续处理
        html_str = html_bytes.decode("utf-8", errors="replace")

    text = extract_plain_text(html_str)

    # Resiliparse 对空白页会返回 ""，此时返回 None 以便调用方判断
    return text if text else None
