"""Extraction — ``resiliparse`` ``main_content`` with section structure.

Section structure (heading hierarchy + char offsets) is returned alongside the
text because breadcrumbs and section vectors both need it.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from vesta.zim.extract import _extract_pdf, extract_article, extract_entry
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
