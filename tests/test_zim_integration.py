"""Real-ZIM integration checks (02-zim-data-layer.md Definition of Done).

These run against the real archives in ``data/zims/`` (gitignored — see
``phased_plan/test-data.md``). They SKIP when the files are absent so CI on a
clean clone still passes; on a dev machine with the corpora dropped in they
verify the headline DoD items that the tiny fixture cannot:

* probed ``has_fulltext_index`` + correct ``Counter['text/html']`` count;
* the preprocessing ladder rescuing a natural-language question, with the
  chosen rung recorded in the trace;
* the WebP-as-``.jpg`` MIME trap on a maxi archive;
* extraction throughput within a sane bound (~1.2 ms/article).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from libzim.reader import Archive

from vesta.config import QUERY_STOPWORDS_LIST
from vesta.retrieval.trace import Trace
from vesta.zim.entries import is_soft_redirect
from vesta.zim.extract import extract_article
from vesta.zim.query import QueryPreparer
from vesta.zim.reader import read_entry_sync
from vesta.zim.registry import _parse_counter_html, _probe_archive
from vesta.zim.search import NoFulltextIndex, fulltext_search, title_suggest

DATA = Path("data/zims")
NOPIC = DATA / "wikipedia_en_top_nopic_2026-06.zim"
MAXI = DATA / "wikipedia_en_100_maxi_2026-07.zim"

nopic = pytest.mark.skipif(not NOPIC.exists(), reason=f"{NOPIC} not present")
maxi = pytest.mark.skipif(not MAXI.exists(), reason=f"{MAXI} not present")


@nopic
def test_nopic_probes_fulltext_and_correct_article_count() -> None:
    """DoD: probed has_fulltext_index + Counter['text/html'] (not over-count)."""
    archive, probe = _probe_archive(NOPIC)
    assert probe.has_fulltext_index is True  # PROBED, not the catalog tag
    # Counter['text/html'] is the correct count; archive.article_count includes
    # redirects and over-counts ~40-75% (measured 875 265 vs 223 149 here).
    assert probe.article_count == _parse_counter_html(archive)
    assert probe.article_count < archive.article_count


@nopic
async def test_natural_language_question_rescued_by_ladder() -> None:
    """DoD: "how do i mount a usb drive" returns hits; trace shows the rung.

    The zero-result trap (raw all-terms AND → 0 for a natural-language
    question) fires on short/trimmed corpora — Simple Wikipedia and the old
    ``mini`` dev archive. On the full-text ``nopic`` archive the question's
    terms co-occur in real articles, so the raw rung matches directly; the DoD
    only requires hits *through the ladder* with the producing rung recorded
    in the trace. The trap mechanics themselves are covered deterministically
    in ``tests/test_zim_query.py``.
    """
    archive = Archive(str(NOPIC))
    qp = QueryPreparer.from_settings(
        stopword_stripping=True,
        stopword_list=QUERY_STOPWORDS_LIST.default,
        ladder_enabled=True,
    )

    async def search(terms: tuple[str, ...], limit: int) -> list[str]:
        try:
            return fulltext_search(archive, terms, limit)
        except NoFulltextIndex:
            return []

    async def suggest(prefix: str, limit: int) -> list[str]:
        return title_suggest(archive, prefix, limit)

    trace = Trace()
    hits = await qp.execute("how do i mount a usb drive", search, suggest, limit=10, trace=trace)
    assert hits, "the NL question must return hits through the ladder"
    assert "USB_flash_drive" in hits, "the question must resolve to the USB article"
    stage = trace.to_dict()["stages"][0]
    chosen = stage["outputs"]["chosen_rung"]
    assert chosen is not None, "the trace must record which rung produced hits"
    assert stage["outputs"][f"rung.{chosen}.hits"] >= 1, "the chosen rung must report its hits"


@maxi
def test_maxi_webp_asset_serves_true_mime() -> None:
    """DoD: a .jpg-named asset is actually WebP and libzim reports image/webp."""
    archive = Archive(str(MAXI))
    webp = None
    for i in range(archive.entry_count):
        entry = archive._get_entry_by_id(i)  # documented iteration API
        if entry.is_redirect:
            continue
        try:
            item = entry.get_item()
        except Exception:
            continue
        if entry.path.endswith((".jpg", ".png")) and item.mimetype == "image/webp":
            webp = entry.path
            break
    assert webp, "maxi archive should contain a .jpg-named WebP asset"
    raw = read_entry_sync(archive, webp)
    assert raw.mimetype == "image/webp"  # serve libzim's type, never the extension
    assert raw.content[:4] == b"RIFF" and raw.content[8:12] == b"WEBP"


def _resolve_relative(article_path: str, rel: str) -> str:
    """Resolve a ZIM-relative link (``../foo``) against the article's directory.

    The reader route mirrors the ZIM path verbatim, so the browser does
    exactly this resolution against the article's base URL — no rewriting needed.
    """
    base_dir = article_path.rsplit("/", 1)[0] + "/" if "/" in article_path else ""
    out: list[str] = []
    for part in (base_dir + rel).split("/"):
        if part == "..":
            if out:
                out.pop()
        elif part not in ("", "."):
            out.append(part)
    return "/".join(out)


@maxi
def test_maxi_nested_article_internal_links_resolve() -> None:
    """DoD: reader is path-preserving on a real nested (maxi) archive.

    The maxi archive uses nested paths (article ``HIV/AIDS``; assets at
    ``../_assets_/...``) — a ``../../`` shape. Because the
    route mirrors the ZIM path and serves bytes unmodified, relative links MUST
    resolve to real archive entries with zero rewriting. We verify the
    server-side precondition: every sampled relative link's resolved target
    exists in the archive. (The browser-side sandbox render is the one
    human-verification gap.)
    """
    import re
    import urllib.parse

    archive = Archive(str(MAXI))
    # Pick an HTML article that actually uses nested relative asset links.
    article_path = None
    html = b""
    for i in range(archive.entry_count):
        entry = archive._get_entry_by_id(i)  # documented iteration API
        if entry.is_redirect:
            continue
        try:
            item = entry.get_item()
        except Exception:
            continue
        if not item.mimetype.startswith("text/html"):
            continue
        content = bytes(item.content)
        if b"<img" in content and b'href="../' in content:
            article_path = entry.path
            html = content
            break
    assert article_path, "maxi archive should have a nested-path article with images"

    raw = read_entry_sync(archive, article_path)
    assert raw.mimetype.startswith("text/html")  # correct content-type, no rewriting
    assert raw.is_redirect is False and raw.soft_redirect_target is None

    # Resolve every relative link against the article's directory; each target
    # must exist in the archive — that is what "internal links resolve" means
    # under a path-preserving passthrough.
    links = [
        urllib.parse.unquote(m.decode("utf-8", "replace").split("#", 1)[0].split("?", 1)[0])
        for m in re.findall(rb'(?:href|src)="([^"]+)"', html)
        if not m.startswith((b"http", b"//", b"#", b"mailto", b"data:"))
    ]
    assert links, "nested article must carry relative links to exercise resolution"
    checked = 0
    for rel in links[:60]:
        target = _resolve_relative(article_path, rel)
        if not target:
            continue
        assert archive.has_entry_by_path(target), (
            f"path-preserving resolution failed: {rel!r} -> {target!r} not in archive"
        )
        checked += 1
    assert checked >= 20, f"expected ≥20 resolvable links; checked {checked}"


@nopic
def test_extraction_throughput_within_sane_bound() -> None:
    """DoD: record extraction throughput (~1.2 ms/article expected).

    Asserts only a SANE upper bound (not the exact projection) so the test is
    robust to slower CI boxes while still catching a regression to
    trafilatura-class slowness (13x). The measured number is printed for the
    report.
    """
    import time

    archive = Archive(str(NOPIC))
    paths: list[str] = []
    for i in range(archive.entry_count):
        if len(paths) >= 400:
            break
        entry = archive._get_entry_by_id(i)  # documented iteration API
        if entry.is_redirect:
            continue
        try:
            item = entry.get_item()
        except Exception:
            continue
        if not item.mimetype.startswith("text/html"):
            continue
        if is_soft_redirect(bytes(item.content)):
            continue
        paths.append(entry.path)

    assert len(paths) >= 100
    start = time.perf_counter()
    for path in paths:
        raw = read_entry_sync(archive, path)
        extract_article(raw.content, path=path, title=raw.title)
    elapsed = time.perf_counter() - start
    per_article_ms = (elapsed / len(paths)) * 1000
    docs_per_s = len(paths) / elapsed
    print(
        f"\n[extraction] {len(paths)} nopic articles in {elapsed * 1000:.0f} ms "
        f"= {docs_per_s:.0f} docs/s, {per_article_ms:.2f} ms/article"
    )
    # Sane bound: research projects ~1.2 ms/article; allow a generous 10x margin
    # for a slow/throttled CI box while still rejecting a 13x trafilatura slip.
    assert per_article_ms < 12.0, f"extraction too slow: {per_article_ms:.2f} ms/article"
