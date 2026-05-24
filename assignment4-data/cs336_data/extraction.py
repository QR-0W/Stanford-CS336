from resiliparse.extract.html2text import extract_plain_text


def extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    for encoding in ("utf-8", "latin-1", "cp1252", "iso-8859-1"):
        try:
            html_str = html_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        html_str = html_bytes.decode("utf-8", errors="replace")

    text = extract_plain_text(html_str)

    return text if text else None
