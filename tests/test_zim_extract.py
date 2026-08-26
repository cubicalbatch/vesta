"""Extraction — ``resiliparse`` ``main_content`` with section structure.

Section structure (heading hierarchy + char offsets) is returned alongside the
text because breadcrumbs and section vectors both need it.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from vesta.zim.extract import _extract_pdf, decode_text, extract_article, extract_entry
from vesta.zim.types import RawEntry

_SAMPLE_HTML = """<html><body>
<h1>Albert Einstein</h1>
<p>Albert Einstein was a German-born theoretical physicist.</p>
<h2>Early life</h2>
<p>He was born in Ulm, in the Kingdom of Württemberg.</p>
<h2>Scientific career</h2>
<p>In 1905 he published four groundbreaking papers.</p>
</body></html>"""


def test_extracts_plain_text_from_html() -> None:
    html = "<html><body><h1>X</h1><p>Hello <b>cruel</b> world.</p></body></html>"
    article = extract_article(html.encode(), path="A/X", title="X")
    # Inline text is NOT shredded (the reason to use resiliparse, not
    # selectolax): "cruel" stays inline with its sentence.
    assert "Hello cruel world" in article.text


def test_sections_partition_text_and_carry_headings() -> None:
    article = extract_article(
        _SAMPLE_HTML.encode(), path="A/Albert_Einstein", title="Albert Einstein"
    )
    text = article.text
    assert text.strip(), "sample article must extract non-empty text"
    sections = article.sections
    assert len(sections) >= 2  # lead + at least one H2 section

    # Sections are contiguous, ordered, cover the whole text, non-empty.
    assert sections[0].char_start == 0
    assert sections[-1].char_end == len(text)
    for a, b in pairwise(sections):
        assert a.char_end <= b.char_start

    # At least one H2 heading appears in the breadcrumb paths.
    h2s = [s for s in sections if s.level >= 2]
    assert h2s, "sample has <h2> headings — they must surface"
    assert any(len(s.heading_path) >= 1 for s in h2s)

    # Section offsets index into text.
    for s in sections:
        assert 0 <= s.char_start <= s.char_end <= len(text)


# ── extract_entry: mimetype-aware dispatch (media/SPA ZIM sidecars) ──────────


def _raw(mimetype: str, content: bytes, *, path: str = "x", title: str = "x") -> RawEntry:
    return RawEntry(
        path=path,
        title=title,
        mimetype=mimetype,
        content=content,
        is_redirect=False,
        redirect_target=None,
        soft_redirect_target=None,
    )


@pytest.mark.parametrize(
    ("mimetype", "content", "expected_needles", "unexpected_needles"),
    [
        (
            "text/vtt",
            b"WEBVTT\n\n00:00:00.000 --> 00:01:39.000\nIntro\n",
            ["Intro"],
            ["WEBVTT", "00:00", "-->"],
        ),
        (
            "text/plain",
            b"a   b\n   c\n",
            ["a b", "c"],
            [],
        ),
        (
            "text/markdown",
            b"# Title\n**bold** [t](u)\n",
            ["Title", "bold"],
            ["**", "[](u)", "[t]"],
        ),
        (
            "text/html",
            b"<html><body><h1>X</h1><p>Hello world.</p></body></html>",
            ["Hello world"],
            [],
        ),
        (
            "application/json",
            b'{"k": 1}',
            [],
            [],
        ),
    ],
)
def test_extract_entry_mimetype_dispatch(
    mimetype: str,
    content: bytes,
    expected_needles: list[str],
    unexpected_needles: list[str],
) -> None:
    article = extract_entry(_raw(mimetype, content, title="intro"))
    if mimetype == "application/json":
        assert article.text == ""
    for needle in expected_needles:
        assert needle in article.text
    for needle in unexpected_needles:
        assert needle not in article.text


def test_extract_entry_non_html_single_implicit_section() -> None:
    article = extract_entry(_raw("text/plain", b"one two three"))
    assert len(article.sections) == 1
    assert article.sections[0].char_start == 0
    assert article.sections[0].char_end == len(article.text)


# ── extract_entry: application/pdf dispatch (nautiluszim document libraries) ──


def _make_text_pdf(text: str) -> bytes:
    """Build a minimal valid one-page text PDF whose content stream shows ``text``.

    Extraction is read-only, so the fixture is a hand-rolled but structurally
    valid PDF (correct xref offsets) — the smallest text PDF pdfium will parse.
    """
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
    ]
    safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = b"BT /F1 12 Tf 72 720 Td (" + safe.encode("latin-1") + b") Tj ET"
    objects.append(
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"
    )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    pdf = b"%PDF-1.4\n"
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref_pos = len(pdf)
    n = len(objects) + 1
    pdf += f"xref\n0 {n}\n".encode()
    pdf += b"0000000000 65535 f \n"
    for off in offsets:
        pdf += f"{off:010d} 00000 n \n".encode()
    pdf += b"trailer\n<< /Size " + str(n).encode() + b" /Root 1 0 R >>\n"
    pdf += b"startxref\n" + str(xref_pos).encode() + b"\n%%EOF\n"
    return pdf


def test_extract_pdf_yields_text_from_synthesized_pdf() -> None:
    """``_extract_pdf`` on a text PDF returns the visible text (the OCR boundary
    is an empty result — image-only scans; a text PDF must yield non-empty)."""
    text = _extract_pdf(_make_text_pdf("Distillation For Home Water Treatment"))
    assert "Distillation" in text
    assert "Water Treatment" in text


def test_extract_entry_pdf_dispatch_yields_single_section() -> None:
    """``application/pdf`` routes through ``_extract_pdf`` and, like the other
    non-HTML text sources, gets one implicit lead section."""
    article = extract_entry(
        _raw(
            "application/pdf",
            _make_text_pdf("Giardia Drinking Water Factsheet"),
            path="files/Water (3).pdf",
            title="Water (3).pdf",
        )
    )
    assert "Giardia" in article.text
    assert len(article.sections) == 1
    assert article.sections[0].char_start == 0
    assert article.sections[0].char_end == len(article.text)


def test_extract_pdf_garbage_or_empty_returns_empty_no_crash() -> None:
    """Corrupt/non-PDF bytes are the OCR-boundary's defensive twin: never crash,
    return ``""`` so the entry drops out of the index at the keep-filter."""
    assert _extract_pdf(b"not a pdf at all") == ""
    assert _extract_pdf(b"") == ""


# ── Charset sniffing & decoding (AUDIT_0824 Z2) ──────────────────────────────


def test_decode_text_empty() -> None:
    assert decode_text(b"") == ""


def test_decode_text_utf8() -> None:
    text = "Hello world! café ñ 日本語 中文"
    assert decode_text(text.encode("utf-8")) == text


def test_decode_text_utf8_with_bom() -> None:
    content = b"\xef\xbb\xbfHello with UTF-8 BOM"
    assert decode_text(content) == "Hello with UTF-8 BOM"


def test_decode_text_utf16_le_and_be_with_bom() -> None:
    text = "Hello UTF-16"
    le_bom = b"\xff\xfe" + text.encode("utf-16-le")
    assert decode_text(le_bom) == text

    be_bom = b"\xfe\xff" + text.encode("utf-16-be")
    assert decode_text(be_bom) == text


def test_decode_text_utf32_le_and_be_with_bom() -> None:
    text = "Hello UTF-32"
    le_32_bom = b"\xff\xfe\x00\x00" + text.encode("utf-32-le")
    assert decode_text(le_32_bom) == text

    be_32_bom = b"\x00\x00\xfe\xff" + text.encode("utf-32-be")
    assert decode_text(be_32_bom) == text


def test_decode_text_html_meta_charset_iso8859_1() -> None:
    html_iso = b'<html><head><meta charset="iso-8859-1"></head><body><h1>Caf\xe9</h1></body></html>'
    assert "Café" in decode_text(html_iso, is_html=True)


def test_decode_text_html_meta_charset_windows_1252() -> None:
    html_cp1252 = (
        b'<html><head><meta charset="windows-1252"></head>'
        b"<body><p>\x93Curly quotes\x94 and \x97 dash</p></body></html>"
    )
    res = decode_text(html_cp1252, is_html=True)
    assert "“Curly quotes”" in res
    assert "—" in res


def test_decode_text_html_http_equiv_content_type() -> None:
    html1 = (
        b"<html><head><meta http-equiv='Content-Type' "
        b"content='text/html; charset=windows-1252'></head><body>\x93Test\x94</body></html>"
    )
    assert "“Test”" in decode_text(html1, is_html=True)

    html2 = (
        b'<html><head><meta http-equiv="Content-Type" '
        b'content="text/html; charset=iso-8859-1"/></head><body>Caf\xe9</body></html>'
    )
    assert "Café" in decode_text(html2, is_html=True)


def test_decode_text_html_xml_declaration() -> None:
    xml = (
        b'<?xml version="1.0" encoding="iso-8859-1"?><html><body><h1>M\xfcnchen</h1></body></html>'
    )
    assert "München" in decode_text(xml, is_html=True)


def test_decode_text_html_asian_encodings() -> None:
    # Shift_JIS
    sjis_text = "日本語テスト".encode("shift_jis")
    sjis_html = (
        b'<html><head><meta charset="shift_jis"></head><body><p>'
        + sjis_text
        + b"</p></body></html>"
    )
    assert "日本語テスト" in decode_text(sjis_html, is_html=True)

    # GBK
    gbk_text = "中文测试".encode("gbk")
    gbk_html = (
        b'<html><head><meta content="text/html; charset=gbk" http-equiv="Content-Type"></head>'
        b"<body><p>" + gbk_text + b"</p></body></html>"
    )
    assert "中文测试" in decode_text(gbk_html, is_html=True)


def test_decode_text_plain_and_markdown_fallback() -> None:
    # Non-HTML text in Windows-1252 (UTF-8 strict fails -> falls back to windows-1252)
    plain_bytes = "Café résumé — naïve".encode("windows-1252")
    assert decode_text(plain_bytes, is_html=False) == "Café résumé — naïve"


def test_decode_text_invalid_charset_falls_back() -> None:
    # Unknown charset falls back to windows-1252 / latin-1
    html_invalid = (
        b'<html><head><meta charset="invalid-codec-123"></head><body><p>Caf\xe9</p></body></html>'
    )
    assert "Café" in decode_text(html_invalid, is_html=True)


def test_extract_article_with_non_utf8_encodings() -> None:
    html_iso = (
        b'<html><head><meta charset="iso-8859-1"></head>'
        b"<body><h1>Caf\xe9 de Flore</h1><p>Situ\xe9 \xe0 Paris.</p></body></html>"
    )
    art = extract_article(html_iso, path="A/Cafe", title="Café")
    assert "Café de Flore" in art.text
    assert "Situé à Paris" in art.text
    assert "\ufffd" not in art.text
    assert any("Café de Flore" in p for s in art.sections for p in s.heading_path)


def test_extract_article_utf8_with_bom() -> None:
    html_bom = b"\xef\xbb\xbf<html><body><h1>BOM Heading</h1><p>BOM Body</p></body></html>"
    art = extract_article(html_bom, path="A/BOM", title="BOM")
    assert "BOM Heading" in art.text
    assert "\ufeff" not in art.text


def test_extract_vtt_with_bom_and_encodings() -> None:
    # UTF-8 with BOM
    vtt_bom = b"\xef\xbb\xbfWEBVTT\n\n00:00:00.000 --> 00:01:00.000\nHello from BOM\n"
    assert extract_entry(_raw("text/vtt", vtt_bom)).text == "Hello from BOM"

    # UTF-16 with BOM
    vtt_u16 = b"\xff\xfe" + "WEBVTT\n\n00:00:00.000 --> 00:01:00.000\nHello UTF-16\n".encode(
        "utf-16-le"
    )
    assert extract_entry(_raw("text/vtt", vtt_u16)).text == "Hello UTF-16"

    # Windows-1252
    vtt_cp1252 = "WEBVTT\n\n00:00:00.000 --> 00:01:00.000\nCafé — intro\n".encode("windows-1252")
    assert extract_entry(_raw("text/vtt", vtt_cp1252)).text == "Café — intro"


def test_extract_plain_and_markdown_non_utf8() -> None:
    plain_iso = "Café au lait\n  Deuxième ligne  \n".encode("iso-8859-1")
    art_plain = extract_entry(_raw("text/plain", plain_iso))
    assert "Café au lait" in art_plain.text
    assert "Deuxième ligne" in art_plain.text
    assert "\ufffd" not in art_plain.text

    md_cp1252 = "# Café\n**“Spécial”**\n".encode("windows-1252")
    art_md = extract_entry(_raw("text/markdown", md_cp1252))
    assert "Café" in art_md.text
    assert "“Spécial”" in art_md.text
    assert "\ufffd" not in art_md.text
