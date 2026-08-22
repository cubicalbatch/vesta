"""Tiny ZIM fixture generator.

Builds a few-hundred-KB ZIM *in test* containing every trap the ZIM research
found, so a regression in any one of them fails a test. Six cases:

1. **Hard redirect** — a real ZIM redirect entry (``is_redirect`` True).
2. **Soft redirect** — an HTML ``<meta refresh>`` page that is *not* flagged
   ``is_redirect``: ~4.7k of these in Simple Wikipedia alone silently
   become near-duplicate junk in the index if not detected.
3. **Disambiguation page** — skipped for indexing, kept readable.
4. **Long multi-section article** — exercises section-aware extraction/passage
   splitting (the breadcrumb in chunking).
5. **WebP-as-``.jpg`` asset** — MIME trap: serve ``item.mimetype``, never
   infer from the extension, or images render broken without ``allow-scripts``.
6. **Nested-path article** — ``A/sub/dir/article``-style deep paths.

Indexing tests assert the fixture *builds* and contains the six cases;
search/extraction tests exercise it end-to-end.
"""

from __future__ import annotations

from pathlib import Path

from libzim import Archive
from libzim.writer import Creator, Hint, Item, StringProvider

# Canonical entry paths the tests look up.
REDIRECT_PATH = "A/USA_Redirect"
REDIRECT_TARGET = "A/United_States"
SOFT_REDIRECT_PATH = "A/USA_Soft"
DISAMBIGUATION_PATH = "A/Mercury"
LONG_ARTICLE_PATH = "A/Albert_Einstein"
WEBP_ASSET_PATH = "I/photo.jpg"
NESTED_PATH = "A/-/nested/deep/article"
# Non-HTML text sidecars (media/SPA ZIM support): the real indexable text of a
# video ZIM lives in these, not its html stubs.
VTT_PATH = "transcripts/intro.vtt"
PLAIN_PATH = "notes/readme.txt"
MARKDOWN_PATH = "notes/notes.md"

#: A minimal valid WebP body (RIFF....WEBP / VP8). Magic bytes are what matters.
WEBP_BYTES = (
    b"RIFF\x24\x00\x00\x00WEBPVP8 \x18\x00\x00\x00\x30\x01\x00\x9d\x01\x2a\x01\x00\x01\x00\x05"
)

#: WebVTT chapter file (the youtube2zim/ted2zim sidecar shape): header, two
#: cue blocks with timestamp ranges and chapter-title text.
VTT_BYTES = (
    b"WEBVTT\n\n"
    b"00:00:00.000 --> 00:01:39.000\nIntro\n\n"
    b"00:01:39.000 --> 00:02:12.000\nPropane Heater\n"
)

PLAIN_BYTES = b"First line of notes.\n   Second line with   extra spaces.\n"

MARKDOWN_BYTES = b"# Title\n\nSome **bold** text and a [link](https://example.com).\n"


class _HtmlItem(Item):
    """A single HTML article in the fixture."""

    def __init__(self, path: str, title: str, html: str) -> None:
        self._path = path
        self._title = title
        self._html = html

    def get_path(self) -> str:
        return self._path

    def get_title(self) -> str:
        return self._title

    def get_mimetype(self) -> str:
        return "text/html"

    def get_hints(self) -> dict[Hint, bool]:
        return {Hint.FRONT_ARTICLE: True}

    def get_contentprovider(self) -> StringProvider:
        return StringProvider(self._html)


class _RawItem(Item):
    """A binary asset (used for the WebP-as-jpg MIME trap)."""

    def __init__(self, path: str, title: str, content: bytes, mimetype: str) -> None:
        self._path = path
        self._title = title
        self._content = content
        self._mimetype = mimetype

    def get_path(self) -> str:
        return self._path

    def get_title(self) -> str:
        return self._title

    def get_mimetype(self) -> str:
        return self._mimetype

    def get_hints(self) -> dict[Hint, bool]:
        return {}

    def get_contentprovider(self) -> StringProvider:
        return StringProvider(self._content)


SOFT_REDIRECT_HTML = (
    "<html><head>"
    # The trap: looks like a normal article, but silently bounces the reader.
    # Modern mwoffliner (2026) targets carry a ``#Section`` fragment — the
    # URL fragment must survive as a fragment, never quoted into the path.
    '<meta http-equiv="refresh" content="0; url=./A/United_States#Government">'
    "<title>USA</title></head>"
    "<body>Redirecting to United States…</body></html>"
)

DISAMBIGUATION_HTML = (
    "<html><head><title>Mercury</title></head>"
    # The MediaWiki disambiguation marker the classifier keys off;
    # without it the page is indistinguishable from an ordinary article.
    '<body class="mw-disambig">'
    "<h1>Mercury</h1><p>Mercury may refer to:</p><ul>"
    '<li><a href="A/Mercury_(planet)">Mercury (planet)</a></li>'
    '<li><a href="A/Mercury_(element)">Mercury (element)</a></li>'
    '<li><a href="A/Mercury_(mythology)">Mercury (mythology)</a></li>'
    "</ul></body></html>"
)


def _long_article_html() -> str:
    sections = "\n".join(
        f"<h2>Section {i}</h2><p>{'Lorem ipsum dolor sit amet. ' * 30}</p>" for i in range(1, 7)
    )
    return (
        "<html><head><title>Albert Einstein</title></head><body>"
        "<h1>Albert Einstein</h1>"
        f"{sections}"
        "</body></html>"
    )


def build_tiny_zim(path: str | Path) -> Path:
    """Write the fixture ZIM to ``path`` and return it."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    with Creator(filename=str(out)) as creator:
        # Required ZIM metadata (libzim enforces a small set).
        creator.add_metadata("Name", "vesta-tiny")
        creator.add_metadata("Title", "Vesta Tiny Test Archive")
        creator.add_metadata("Description", "Six ZIM trap cases for tests")
        creator.add_metadata("Language", "eng")
        creator.add_metadata("Date", "2026-07-30")
        creator.add_metadata("Creator", "vesta-tests")
        creator.add_metadata("Publisher", "vesta")
        creator.set_mainpath(LONG_ARTICLE_PATH)

        # Case 4 first so the redirect target exists before we point at it.
        creator.add_item(_HtmlItem(LONG_ARTICLE_PATH, "Albert Einstein", _long_article_html()))
        # The redirect target must exist as a real article: libzim drops dangling
        # redirects at finalise time, which would silently remove case 1.
        creator.add_item(
            _HtmlItem(
                REDIRECT_TARGET,
                "United States",
                "<html><body><h1>United States</h1><p>A country.</p></body></html>",
            )
        )
        # Case 1: hard redirect.
        creator.add_redirection(REDIRECT_PATH, "USA", REDIRECT_TARGET, {Hint.FRONT_ARTICLE: True})
        # Case 2: soft redirect (meta refresh) — a plain article, NOT is_redirect.
        creator.add_item(_HtmlItem(SOFT_REDIRECT_PATH, "USA", SOFT_REDIRECT_HTML))
        # Case 3: disambiguation page.
        creator.add_item(_HtmlItem(DISAMBIGUATION_PATH, "Mercury", DISAMBIGUATION_HTML))
        # Case 5: asset named .jpg but actually WebP (MIME trap). libzim stores
        # the TRUE mimetype (image/webp) despite the .jpg path — the reader must
        # serve that verbatim, never inferring from the extension. Verified on
        # real maxi archives: .jpg-named assets report image/webp + RIFF…WEBP.
        creator.add_item(_RawItem(WEBP_ASSET_PATH, "photo", WEBP_BYTES, "image/webp"))
        # Case 6: nested-path article.
        creator.add_item(_HtmlItem(NESTED_PATH, "Nested Article", "<html><body>deep</body></html>"))
        # Non-HTML text sidecars (media/SPA ZIM support): vtt chapters, plain
        # notes, markdown — the formats whose real text the mimetype-aware
        # extractor harvests. Assets (no FRONT_ARTICLE hint) like the WebP above.
        creator.add_item(_RawItem(VTT_PATH, "intro transcript", VTT_BYTES, "text/vtt"))
        creator.add_item(_RawItem(PLAIN_PATH, "readme", PLAIN_BYTES, "text/plain"))
        creator.add_item(_RawItem(MARKDOWN_PATH, "notes", MARKDOWN_BYTES, "text/markdown"))
    return out


def open_tiny_zim(path: str | Path) -> Archive:
    """Open the fixture for read-back assertions."""
    return Archive(str(path))
