"""HTML → section-aware plain text via ``resiliparse``.

Why ``resiliparse`` and not the alternatives:

* ``selectolax`` matches the speed (~45 MB/s) but **shreds sentence
  boundaries** — ``'was a\\nPortuguese\\n\\nneurologist'`` — which silently
  poisons embeddings and LLM context alike. Disqualified.
* ``trafilatura`` is correct but 13x slower (~127 docs/s).
* ``resiliparse`` ``main_content`` mode hits ~45 MB/s / ~1 700 docs/s with
  *correct* inline handling — ``'was a Portuguese neurologist.'`` — and built-in
  boilerplate removal.

Section structure (heading hierarchy + char offsets into the extracted text) is
returned alongside the text because passage breadcrumbs and section vectors
both need it. Offsets are computed by walking the same parsed tree resiliparse
extracts from, then locating each heading's text in the extracted output — so the
two stay aligned by construction.

Bulk extraction must be **multi-process, not multi-threaded** (threads scale
negatively under GIL contention, whereas processes scale linearly).
``extract_many`` provides the multi-process bulk helper; single-article
query-time reads use the inline ``extract_article``.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import re
from collections.abc import Sequence
from typing import Any

from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.html import HTMLTree

from vesta.zim.entries import STUB_CHAR_THRESHOLD
from vesta.zim.types import EntryFlags, EntryPath, ExtractedArticle, RawEntry, Section

_log = logging.getLogger(__name__)

#: ``pypdfium2`` (BSD-3, native PDFium) document type, lazily resolved at
#: module load. ``None`` when the optional dep is absent — ``_extract_pdf``
#: then degrades to empty text rather than crashing. Native code is fine
#: under a permissive licence (PDFium: BSD-3); AGPL extractors (PyMuPDF)
#: are excluded on purpose. A PDF that yields no text is an image-only
#: scan → the OCR boundary (reported, not worked around here).
_PdfDocument: Any = None
try:
    from pypdfium2 import PdfDocument as _pdfium_PdfDocument

    _PdfDocument = _pdfium_PdfDocument
except ImportError:  # optional dep absent; PDF extraction degrades to empty text
    pass

#: Boilerplate resiliparse's ``main_content`` does not reliably remove; dropped
#: before extraction so navboxes/reference-apparatus link soup never reaches the
#: text. Infoboxes are deliberately kept (dense factual content that retrieves
#: well even if it embeds poorly — keeping vs excluding is a retrieval/indexing
#: policy decision, not an extraction one).
_STRIP_SELECTORS = (
    "script",
    "style",
    "table.navbox",
    ".navbox",
    "ol.references",
    ".references",
    ".reflist",
    ".refbegin",
    ".mw-references-wrap",
    ".reference",
    "sup.reference",
    ".mw-editsection",
    ".hatnote",
    ".noprint",
    ".metadata",
    ".mbox",
    ".ambox",
    ".mw-empty-elt",
    ".shortdescription",
    # "See also" / "External links" / "References" appendix sections.
    ".seealso",
    ".see-also",
    ".external-links",
    ".external-Links",
)

_HEADINGS = ("h1", "h2", "h3", "h4", "h5", "h6")


def _strip_boilerplate(tree: HTMLTree) -> None:
    """Remove navboxes / reference lists / appendix sections in place.

    Each selector is attempted independently and tolerated if Lexbor rejects
    it — fixtures and non-MediaWiki scrapers won't have these classes.
    """
    body = tree.body
    for selector in _STRIP_SELECTORS:
        try:
            for node in body.query_selector_all(selector):
                node.decompose()
        except Exception:  # unsupported selector on this parser
            continue


def _collect_headings(body: object) -> list[tuple[int, str]]:
    """Headings ``(level, text)`` in document order (recursive descent).

    ``child_nodes`` yields only elements (resiliparse hides raw text nodes),
    which is exactly what we want — we only ever classify element tags here.
    """
    headings: list[tuple[int, str]] = []

    def walk(node: object) -> None:
        for child in getattr(node, "child_nodes", []):
            tag = str(getattr(child, "tag", "")).lower()
            if tag in _HEADINGS:
                text = str(getattr(child, "text", "")).strip()
                if text:
                    headings.append((int(tag[1]), text))
            walk(child)

    walk(body)
    return headings


def _build_sections(text: str, headings: Sequence[tuple[int, str]]) -> tuple[Section, ...]:
    """Map headings to char ranges in ``text`` and return the section partition.

    Each heading's text is located from a moving cursor (headings appear in
    order, so a forward search from the previous match is monotonic and robust).
    Sections are contiguous and cover the whole text. A heading whose text
    isn't found (boilerplate removal dropped it) extends the previous section.
    """
    if not headings:
        # One implicit section spanning the whole article (the lead).
        return (Section(heading_path=(), level=0, char_start=0, char_end=len(text)),)

    boundaries: list[tuple[int, tuple[str, ...], int]] = []  # (char_start, path, level)
    stack: list[tuple[int, str]] = []  # (level, title)
    cursor = 0
    for level, title in headings:
        idx = text.find(title, cursor)
        if idx < 0:
            # Heading text not in extracted output (stripped/mangled): still use
            # it for the breadcrumb path, but don't start a new char range — fold
            # into the current position so sections stay contiguous & non-empty.
            char_start = cursor
        else:
            char_start = idx
            cursor = idx + len(title)
        # Maintain the heading hierarchy stack for the breadcrumb path.
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        boundaries.append((char_start, tuple(t for _, t in stack), level))

    # Close each section at the next boundary's start; last runs to end of text.
    sections: list[Section] = []
    for i, (start, path, level) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        # Collapse accidental zero/negative-width sections onto the next.
        end = max(end, start)
        sections.append(Section(heading_path=path, level=level, char_start=start, char_end=end))
    return tuple(sections)


def extract_article(html_bytes: bytes, *, path: EntryPath, title: str) -> ExtractedArticle:
    """Extract one article: text + section structure + stub flag.

    Synchronous and CPU-bound (~2-6 ms/article measured on Wikipedia-sized
    HTML, 2026-08-20; scales with document size); the registry dispatches it
    through a bounded pool so it never blocks the event loop. This is the
    *HTML* path — see :func:`_extract_pdf` for the one that is three orders
    of magnitude slower.
    """
    html = html_bytes.decode("utf-8", "replace")
    tree = HTMLTree.parse(html)
    _strip_boilerplate(tree)
    cleaned_html = tree.body.html
    text = extract_plain_text(cleaned_html, main_content=True, alt_texts=False, links=False)
    headings = _collect_headings(tree.body)
    sections = _build_sections(text, headings)
    flags = EntryFlags.STUB if len(text) < STUB_CHAR_THRESHOLD else EntryFlags.NONE
    article_title = title or (tree.title or path)
    return ExtractedArticle(
        path=path,
        title=article_title,
        text=text,
        sections=sections,
        flags=flags,
    )


# ── Mimetype-aware extraction (media/SPA ZIM sidecars) ───────────────────────
#
# ``extract_entry`` dispatches on the entry's true libzim mimetype. ``text/html``
# goes through resiliparse above; ``text/vtt`` / ``text/plain`` / ``text/markdown``
# are harvested as plain text (a media ZIM's transcripts/chapters/notes — the
# real indexable content, since its ``text/html`` entries are redirect stubs).
# This runs at BOTH index time (indexer worker) and query time
# (``LocalArchive.extract``), keyed off the live mimetype, so a chunk's
# ``char_start``/``char_end`` recover consistently with no mimetype column.

#: A standalone VTT timecode (``00:00:00.000`` or ``00:01:39``), used to drop
#: cue-timestamp lines that lack the ``-->`` arrow.
_VTT_TIMECODE_RE = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?(\.\d+)?$")


def _extract_vtt(content: bytes) -> str:
    """WebVTT → plain text: drop the header, NOTE blocks, cue timestamps and
    standalone timecodes; keep cue text (the transcript / chapter titles).

    VTT is the subtitle/chapter format video scrapers (youtube2zim, ted2zim)
    emit. Format: ``WEBVTT`` header, then repeated ``[cue-id]\\n<time> --> <time>
    \\n<text>`` blocks. The cue-id line is optional and not always present; we
    keep every line that isn't structurally a timestamp/header/note."""
    text = content.decode("utf-8", "replace")
    kept: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        upper = line.upper()
        if upper == "WEBVTT" or upper.startswith("WEBVTT "):
            continue
        if upper.startswith("NOTE"):
            continue
        if "-->" in line:  # the cue timestamp range line
            continue
        if _VTT_TIMECODE_RE.match(line):  # a stray standalone timecode
            continue
        kept.append(line)
    return "\n".join(kept)


def _extract_plain(content: bytes) -> str:
    """``text/plain`` → text with per-line whitespace normalised."""
    lines = [
        re.sub(r"[ \t]+", " ", ln).strip() for ln in content.decode("utf-8", "replace").splitlines()
    ]
    return "\n".join(ln for ln in lines if ln)


def _extract_markdown(content: bytes) -> str:
    """``text/markdown`` → text with common markup stripped (ATX heading marks,
    emphasis, inline-code, link URLs). One passage per format — heading-aware
    section splitting is not attempted, so depth 2 collapses to the lead."""
    out: list[str] = []
    for raw_line in content.decode("utf-8", "replace").splitlines():
        line = raw_line.strip()
        if not line:
            out.append("")
            continue
        line = re.sub(r"^#{1,6}\s+", "", line)  # leading ATX heading marks
        line = line.replace("**", "").replace("__", "").replace("`", "")
        line = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", line)  # [text](url) → text
        out.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def _extract_pdf(content: bytes) -> str:
    """``application/pdf`` → plain text via ``pypdfium2`` (native PDFium).

    Returns ``""`` for image-only scans, encrypted/corrupt PDFs, or when the
    optional dep is absent — this is the OCR boundary; an empty result drops
    out of the index at the keep-filter rather than crashing. Runs
    identically at index time (worker) and query time (``LocalArchive.extract``)
    keyed off the live mimetype.

    pdfium is native and linear-ish in page count: measured 2026-08-20 on
    ``zimgit-water_en``, 0.18 s for the 8 MB / worst case and 0.41 s for all
    7 PDFs — where pdfminer.six (pure Python, the previous extractor) took
    8.96 s and 13.7 s (33x slower; A/B recorded in
    ``phased_plan/24-candidate-articles.findings.md``). Query-time
    extraction is uncached by design, so a PDF nominated by the funnel
    costs this on *every* query — keep it cheap before reaching for a cache.
    """
    if _PdfDocument is None:
        return ""
    try:
        doc = _PdfDocument(content)
        try:
            parts: list[str] = []
            for i in range(len(doc)):
                page = doc[i]
                try:
                    textpage = page.get_textpage()
                    try:
                        parts.append(textpage.get_text_range() or "")
                    finally:
                        textpage.close()
                finally:
                    page.close()
            return "\n".join(parts)
        finally:
            doc.close()
    except Exception as exc:  # a corrupt/encrypted PDF never aborts extraction
        _log.warning("zim.pdf_extractFailed error=%r", exc)
        return ""


def _single_section(text: str) -> tuple[Section, ...]:
    """One implicit lead section spanning the whole text (no heading structure)."""
    return (Section(heading_path=(), level=0, char_start=0, char_end=len(text)),)


def extract_entry(raw: RawEntry) -> ExtractedArticle:
    """Mimetype-aware extraction: HTML → resiliparse; vtt/plain/markdown/pdf →
    plain text with a single implicit lead section.

    HTML soft/hard redirects are handled upstream by ``LocalArchive.extract``
    (which short-circuits to empty text); this function only runs for entries
    that survived that check, so ``raw`` is a real content entry.
    """
    mime = (raw.mimetype or "").lower()
    title = raw.title or raw.path
    if mime.startswith("text/html"):
        return extract_article(raw.content, path=raw.path, title=title)
    if mime.startswith("text/vtt"):
        text = _extract_vtt(raw.content)
    elif mime.startswith("text/plain"):
        text = _extract_plain(raw.content)
    elif mime.startswith("text/markdown"):
        text = _extract_markdown(raw.content)
    elif mime.startswith("application/pdf"):
        text = _extract_pdf(raw.content)
    else:
        # Not a text-bearing entry — nothing to index; empty text drops out at
        # the keep-filter.
        return ExtractedArticle(
            path=raw.path, title=title, text="", sections=(), flags=EntryFlags.NONE
        )
    flags = EntryFlags.NONE
    if len(text) < STUB_CHAR_THRESHOLD:
        flags |= EntryFlags.STUB
    return ExtractedArticle(
        path=raw.path,
        title=title,
        text=text,
        sections=_single_section(text),
        flags=flags,
    )


# ── Multi-process bulk helper ────────────────────────────────────────────────
# libzim's ``Archive`` is not picklable; each worker opens its own from the
# path in a Pool initializer. Threads scale negatively for this workload, so
# bulk extraction is multi-process by construction.


def _mp_init(archive_path: str) -> None:
    """Pool initializer: open the archive once per worker process."""
    from libzim.reader import Archive as LibzimArchive

    global _MP_ARCHIVE
    _MP_ARCHIVE = LibzimArchive(archive_path)


_MP_ARCHIVE: object | None = None


def _mp_extract(path: EntryPath) -> ExtractedArticle:
    """Pool worker: read + extract one path. Assumes ``_mp_init`` ran."""
    assert _MP_ARCHIVE is not None  # invariant of the initializer
    from vesta.zim.reader import read_entry_sync

    raw = read_entry_sync(_MP_ARCHIVE, path)
    return extract_article(raw.content, path=path, title=raw.title)


def extract_many(
    archive_path: str,
    paths: Sequence[EntryPath],
    *,
    processes: int = 1,
) -> list[ExtractedArticle]:
    """Bulk multi-process extraction. Thin by design.

    ``processes`` defaults to 1; callers pass an explicit count (the CLI
    benchmark hardcodes its measurement point). Returns results in ``paths``
    order. An empty ``paths`` is a no-op (avoids spinning up a Pool for
    nothing).
    """
    if not paths:
        return []
    if processes <= 1:
        # Inline fallback keeps tests deterministic without process spawn.
        from libzim.reader import Archive as LibzimArchive

        from vesta.zim.reader import read_entry_sync

        archive = LibzimArchive(archive_path)
        out: list[ExtractedArticle] = []
        for p in paths:
            raw = read_entry_sync(archive, p)
            out.append(extract_article(raw.content, path=p, title=raw.title))
        return out
    with mp.get_context("spawn").Pool(
        processes=processes, initializer=_mp_init, initargs=(archive_path,)
    ) as pool:
        return pool.map(_mp_extract, list(paths))


__all__ = ["extract_article", "extract_entry", "extract_many"]
