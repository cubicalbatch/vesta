"""Candidate-articles passage builder — read, extract, split to passages.

Registered as ``passage_builder`` ``candidate_articles``. For each fused candidate,
reads the full article text from the ZIM (mmap, ~ms), extracts it via
``resiliparse``, and splits into ~400-token sentence-aligned passages via
``zim/passages.py``. Capped at ``max_articles`` top candidates and
``max_passages`` total passages.

Speed: dominant cost is extraction, and it is **mimetype-dependent by orders
of magnitude**. HTML via resiliparse is ~2-6 ms/article; ``application/pdf``
via pypdfium2 (native PDFium) is ~0.2 s for an 8 MB
document — linear-ish in page count. The cap at ``max_articles`` bounds the
*count*, not the cost. Measured 2026-08-20 on the 9-archive corpus: with
pdfminer.six (the pure-Python extractor this replaced) an unscoped level-3
query spent 14.2 s here, 99.2 % of it four PDF candidates from one
7-document archive; after the swap the whole stage is sub-second.

Extraction is **not cached** — the same document is re-extracted on every
query that nominates it. That was ruinous under pdfminer and is acceptable
under pdfium; it is still the first thing to revisit if this stage shows up
in a latency profile again.

Requires no capabilities — reads are always available from the ZIM.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel

from vesta.config.capabilities import Capability
from vesta.retrieval.contracts import Candidate, PreparedQuery
from vesta.retrieval.registry import register
from vesta.zim.types import EntryPath, Passage

if TYPE_CHECKING:
    from vesta.retrieval.trace import Trace
    from vesta.zim.registry import ArchiveRegistry


def _title_passage(zim_id: int, path: EntryPath, title: str) -> Passage:
    """A synthesized passage whose text IS the article title.

    Used when a candidate's body extracts to empty (a media/SPA-ZIM
    meta-refresh stub, a gallery entry, a soft redirect). Without this the
    candidate is silently dropped even though a source matched it; with it the
    candidate surfaces as a title card. Generic — never branches on ZIM kind.

    ``char_start``/``char_end`` are 0 (there is no underlying article.text to
    index into); the passage is query-time-only and self-contained, so the
    offset-recovery invariant only matters for indexed chunks, which these
    candidates have none of.
    """
    return Passage(
        zim_id=zim_id,
        path=path,
        ordinal=0,
        char_start=0,
        char_end=0,
        breadcrumb=title,
        text=title,
        is_lead=True,
    )


@register("passage_builder", "candidate_articles")
class CandidateArticles:
    """Read candidate articles from ZIM and split into passages."""

    requires: ClassVar[frozenset[Capability]] = frozenset()

    class Params(BaseModel):
        max_articles: int = 20
        max_passages: int = 200
        #: When a candidate's body is empty but it has a title, synthesize a
        #: title passage so it still surfaces as a card (default on). Generic
        #: robustness — a retrieval profile can set this false to restore the
        #: pre-2026-08 "drop empty-body candidates" behaviour if a golden-set
        #: A/B shows it regresses a corpus.
        title_fallback: bool = True

    def __init__(
        self, params: Params | None = None, archives: ArchiveRegistry | None = None
    ) -> None:
        self._params = params or self.Params()
        self._archives = archives

    async def build(self, cands: list[Candidate], q: PreparedQuery, tr: Trace) -> list[Passage]:
        """For each candidate (up to ``max_articles``), extract article text and
        split into passages (up to ``max_passages`` total). Order preserves the
        fused candidate ranking."""
        from vesta.zim.passages import split_passages

        if not cands or self._archives is None:
            return []

        max_articles = self._params.max_articles
        max_passages = self._params.max_passages
        title_fallback = self._params.title_fallback

        passages: list[Passage] = []
        seen: set[tuple[int, str]] = set()

        for cand in cands[:max_articles]:
            if len(passages) >= max_passages:
                break
            key = (cand.zim_id, cand.path)
            if key in seen:
                continue
            seen.add(key)

            try:
                archive = self._archives.get(cand.zim_id)
                article = await archive.extract(cand.path)
            except Exception:
                continue

            if article and article.text.strip():
                parts = split_passages(
                    article,
                    target_tokens=400,
                    sentence_aligned=True,
                    breadcrumb_enabled=True,
                    zim_id=cand.zim_id,
                )
            elif title_fallback and article and article.title.strip():
                # Empty body, real title (e.g. a media-ZIM redirect stub a
                # lexical source still matched): surface it as a title passage
                # rather than dropping it. Scorers/snippets/budget treat the
                # title as the text, so every downstream stage works unchanged.
                parts = [_title_passage(cand.zim_id, cand.path, article.title)]
            else:
                continue

            remaining = max_passages - len(passages)
            passages.extend(parts[:remaining])

        return passages
