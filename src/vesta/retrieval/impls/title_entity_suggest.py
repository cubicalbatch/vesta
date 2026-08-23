"""Entity-shaped title lookup candidate source.

Registered as ``candidate_source`` ``title_entity_suggest``. ``title_suggest``
joins every normalized term into one prefix string, so on a natural-language
question it hands the title index a non-prefix like "old napoleon became
emperor" — measured 0/150 gold articles. This source gives the
universal ``X/title/xapian`` index what it can actually prefix-match: entity-
shaped spans lifted from the **raw** question string.

Spans come from ``q.raw``, never ``q.terms``: casing is the signal, and the
``normalize`` preparer has already lowercased it away ("casing is
signal"). Three extraction tiers, strongest first:

1. capitalized runs — ``Elizabeth II``, ``Alexander Graham Bell``,
   ``FIFA World Cup``, bridged particles included (``University of Toronto``,
   ``Vincent van Gogh``); a sentence-opening capitalized word is only dropped
   when the run is a single known sentence opener (``How``, ``In``, ``The``),
   never chopped out of a multi-word run — the prototype that reduced
   ``Elizabeth II`` to ``II`` is the documented failure this fixes. A single
   capitalized word is only emitted as a fallback when the question names no
   multi-word entity at all (``How many moons orbit Jupiter?``) — beside a
   multi-word entity it is almost always a modifier ("the Illinois bar", "the
   Swarovski crystals"), and measured, trusting it loses 3 hybrid questions
   for every 4 it rescues;
2. quoted spans — ``'Nembutal'``, ``"Greater Aryan certificate"`` (titles are
   exactly what users quote);
3. behind ``include_distinctive_terms`` (off by default; the T1/T2
   measurement says distinctive terms lift recall but add generic single-word
   prefixes): distinctive single terms of 4+ chars, non-stopword —
   ``raloxifene``, ``meningioma`` — for questions whose gold article is named
   in lowercase.

One ``Archive.suggest`` per span per archive, capped at ``max_spans`` (4
default, ~3 ms warm each on the pinned ZIM, serialized per archive by the
registry's search lock and gated across sources by the pipeline's central
semaphore). When a span's suggest results contain the span's own article
(exact title, case-insensitively — measured at rank 1-2 whenever it exists),
only that article is emitted: the non-exact tail is variant dilution, and
variant nominations were the three measured full losses on hybrid. Candidates
keep span-major ordering — a hit from a stronger span outranks a hit from a
weaker one — and RRF (``k=20``) fuses them with the other lexical sources per
archive; no fusion change is needed for a third lexical source. Ranks start
at ``rank_offset`` (default 4): under round-robin cross-archive union every
within-archive rank step is amplified by the archive count, so an untempered
title-only nomination can displace an article backed by both xapian and the
dense side — the offset lets this source *support* candidates instead of
outranking them.

Requires no capabilities — like ``title_suggest``, the title index is present
in every archive tested, including archives without full text.
"""

from __future__ import annotations

import asyncio
import re
import string
from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel

from vesta.config.capabilities import Capability
from vesta.retrieval.contracts import Candidate, PreparedQuery, Scope
from vesta.retrieval.registry import register
from vesta.zim.query import DEFAULT_STOPWORDS
from vesta.zim.types import entry_title_key

if TYPE_CHECKING:
    from vesta.retrieval.trace import Trace
    from vesta.zim.registry import ArchiveRegistry
    from vesta.zim.types import Archive

# Curly quotes/apostrophes appear in real queries ("Rupert's drop" typed with
# U+2019); written as escapes because RUF001 flags the literal confusables.
_LQ, _RQ = "\u2018", "\u2019"  # single curly quotes
_LDQ, _RDQ = "\u201c", "\u201d"  # double curly quotes

#: One capitalized token: starts uppercase, then word chars / apostrophes /
#: hyphens (``COVID-19``, ``Rupert's``, ``McClintock``).
_CAP_TOKEN = "[A-Z][\\w'\\u2019-]*"

#: Lowercase particles that legitimately sit *inside* a capitalized entity
#: (``University of Toronto``, ``Vincent van Gogh``, ``Ibn Battuta``). A
#: particle only extends the run when another capitalized token follows.
_CONNECTOR = "(?:of|de|der|den|van|von|di|da|du|del|della|la|le|the|al|el|ibn|bin|mac|mc)"

_CAP_RUN_RE = re.compile(rf"\b{_CAP_TOKEN}(?:\s+{_CAP_TOKEN}|\s+{_CONNECTOR}\s+{_CAP_TOKEN})*")

#: Quoted spans, straight or curly, single or double. Users quote titles. The
#: straight single quote ``'`` is ambiguous — it is also the English apostrophe
#: ("I've", "Scheler's") — so it only counts as an opening quote when it is not
#: preceded by a word char / apostrophe, and as a closing quote when not
#: followed by one. Curly quotes and double quotes are unambiguous.
_QUOTED_RE = re.compile(
    rf'[{_LDQ}"]([^{_LDQ}{_RDQ}"]{{2,80}})[{_RDQ}"]'
    rf"|[{_LQ}]([^{_LQ}{_RQ}]{{2,80}})[{_RQ}]"
    rf"|(?<![\w'\\u2019])'([^']{{2,80}})'(?![\w'\\u2019])"
)

#: Words that only carry their capital from opening a sentence/question
#: ("How old...", "In lampreys...", "The Great Depression..."). A single-token
#: capitalized run that starts a sentence is dropped only when it lowercases
#: into this set; anything else ("Napoleon died...", "COVID-19 cases...") is
#: kept. DEFAULT_STOPWORDS already holds the interrogatives and most openers;
#: the extras are the leading prepositions/conjunctions it lacks plus the
#: contracted pronoun openers the token regex keeps in one piece.
_SENTENCE_OPENERS = frozenset(DEFAULT_STOPWORDS) | frozenset(
    (
        "after",
        "during",
        "besides",
        "despite",
        "following",
        "according",
        "roughly",
        "approximately",
        "considering",
        "imagine",
        "suppose",
        "let's",
        "i've",
        "i'm",
        "i'd",
        "we're",
        "we've",
        "they're",
        "you're",
        "he's",
        "she's",
        "it's",
        "that's",
        "there's",
        "what's",
        "who's",
    )
)

_STOPWORDS = frozenset(DEFAULT_STOPWORDS)

#: Words that carry a capital mid-sentence without naming an entity —
#: demonyms and nationality/period adjectives. Not a junk denylist; a denylist
#: of *known non-entities*, so "the German philosopher Max Scheler" yields
#: "Max Scheler" first and "German" not at all. Multi-token runs
#: ("French Revolution") are never touched by this list.
_NONENTITY_CAPITALS = frozenset(
    (
        "african",
        "american",
        "ancient",
        "arab",
        "asian",
        "australian",
        "austrian",
        "belgian",
        "brazilian",
        "british",
        "canadian",
        "chinese",
        "christian",
        "communist",
        "czech",
        "danish",
        "dutch",
        "egyptian",
        "english",
        "european",
        "french",
        "german",
        "greek",
        "hindu",
        "indian",
        "irish",
        "islamic",
        "israeli",
        "italian",
        "japanese",
        "jewish",
        "korean",
        "latin",
        "medieval",
        "muslim",
        "nazi",
        "norwegian",
        "persian",
        "polish",
        "portuguese",
        "russian",
        "scottish",
        "serbian",
        "soviet",
        "spanish",
        "swedish",
        "swiss",
        "turkish",
        "welsh",
        "western",
    )
)

_PUNCT_STRIP = string.punctuation + _LQ + _RQ + _LDQ + _RDQ + "…€£₹" + " \t"

#: A span longer than this is not title-shaped; suggesting with it would burn
#: one of the ``max_spans`` slots on a prefix the index cannot match.
_MAX_SPAN_CHARS = 60


def _sentence_starts(text: str) -> frozenset[int]:
    """Character offsets where a sentence (so a fresh capitalization) begins."""
    starts = {0}
    for m in re.finditer(r"[.!?][\"'\u2019\u201d)]*\s+", text):
        starts.add(m.end())
    return frozenset(starts)


def _clean_span(span: str) -> str:
    """Strip edge punctuation and a trailing possessive; collapse whitespace."""
    out = " ".join(span.split()).strip(_PUNCT_STRIP)
    if out.lower().endswith("'s") or out.lower().endswith(_RQ + "s"):
        out = out[:-2].strip(_PUNCT_STRIP)
    return out


def _norm_word(word: str) -> str:
    """Lowercase with curly apostrophes folded, for list membership checks."""
    return word.lower().replace(_RQ, "'").replace(_LQ, "'")


def _capitalized_spans(raw: str, starts: frozenset[int]) -> list[tuple[str, int, int]]:
    """Capitalized runs as ``(text, position, strength)``.

    Strength 2 = a multi-token run — a title-shaped entity even at sentence
    start ("Elizabeth II was queen..."). Strength 1 = a single capitalized
    token that survives the sentence-opener and demonym filters.
    """
    out: list[tuple[str, int, int]] = []
    for m in _CAP_RUN_RE.finditer(raw):
        span = _clean_span(m.group(0))
        tokens = span.split()
        if not tokens:
            continue
        # A capital at a sentence start may only be the sentence's opening
        # word. Single-token openers ("How old...", "I've read...") are
        # dropped whole; a multi-token run keeps its entity but sheds the
        # opener ("The Great Depression..." -> "Great Depression",
        # "In Norbert Wiener's..." -> "Norbert Wiener"). Anything else
        # ("Napoleon died...", "Elizabeth II was queen...") is kept intact.
        if m.start() in starts and _norm_word(tokens[0]) in _SENTENCE_OPENERS:
            tokens = tokens[1:]
            span = _clean_span(" ".join(tokens))
        if len(tokens) == 1 and _norm_word(tokens[0]) in _NONENTITY_CAPITALS:
            continue
        if len(span) < 3 or len(span) > _MAX_SPAN_CHARS:
            continue
        if span.lower() in _STOPWORDS:
            continue
        out.append((span, m.start(), 2 if len(tokens) > 1 else 1))
    return out


def _quoted_spans(raw: str) -> list[tuple[str, int, int]]:
    """Quoted substrings as ``(text, position, strength)`` — strength 2."""
    out: list[tuple[str, int, int]] = []
    for m in _QUOTED_RE.finditer(raw):
        inner = next(g for g in m.groups() if g is not None)
        span = _clean_span(inner)
        if len(span) < 3 or len(span) > _MAX_SPAN_CHARS:
            continue
        if span.lower() in _STOPWORDS:
            continue
        out.append((span, m.start(), 2))
    return out


def _distinctive_terms(raw: str) -> list[tuple[str, int, int]]:
    """Distinctive single terms (strength 0): 4+ chars, non-stopword.

    Filtered by the same opener/demonym lists as capitalized spans — a term
    like "During" or "German" is no more a title prefix for being lowercase.
    """
    out: list[tuple[str, int, int]] = []
    seen: set[str] = set()
    for m in re.finditer(r"[^\W]+", raw, re.UNICODE):
        word = m.group(0)
        low = _norm_word(word)
        if (
            len(word) < 4
            or low in _STOPWORDS
            or low in _SENTENCE_OPENERS
            or low in _NONENTITY_CAPITALS
            or low in seen
        ):
            continue
        seen.add(low)
        out.append((word, m.start(), 0))
    return out


def extract_spans(
    raw: str, *, max_spans: int = 4, include_distinctive_terms: bool = False
) -> list[str]:
    """Entity-shaped suggest prefixes from the raw question, strongest first.

    Ordering: strength (multi-token entity / quoted > single capitalized token
    > distinctive term), then reading order. ``max_spans`` caps the total —
    each span costs one ``suggest`` call per archive.

    This is the one place retrieval needs the raw string: casing, quotes, and
    token shape are all destroyed by ``normalize`` ("casing is
    signal").
    """
    if not raw.strip():
        return []
    starts = _sentence_starts(raw)
    ranked = _capitalized_spans(raw, starts) + _quoted_spans(raw)
    if include_distinctive_terms:
        ranked += _distinctive_terms(raw)
    # A single capitalized word is too ambiguous a title prefix when the
    # question also names a multi-word entity ("the Illinois bar" beside
    # "Abraham Lincoln", "the Swarovski crystals" beside "Derek McLane") —
    # measured: single-token spans rescue 4 questions but lose 3 on hybrid by
    # nominating a modifier's article over the gold. They are kept only as a
    # fallback for entity-poor questions ("How many moons orbit Jupiter?"),
    # where they are all the casing signal there is.
    if any(s == 2 for _, _, s in ranked):
        ranked = [(t, p, s) for t, p, s in ranked if s != 1]
    ranked.sort(key=lambda t: (-t[2], t[1]))

    spans: list[str] = []
    seen_spans: set[str] = set()  # exact kept span texts (lowered)
    tokens_of: set[str] = set()  # every token of every kept span (lowered)
    for span, _pos, _strength in ranked:
        low = span.lower()
        if low in seen_spans or low in tokens_of:
            continue  # duplicate, or a bare token of a kept entity span
        spans.append(span)
        seen_spans.add(low)
        tokens_of.update(_norm_word(t) for t in span.split())
        if len(spans) >= max_spans:
            break
    return spans


def _exact_title_hits(span: str, paths: list[str]) -> list[str]:
    """The subset of ``suggest`` results that are the span's own article.

    Normalization (basename, underscores to spaces, lowercase) mirrors
    ``alias_title_resolve._exact_matches`` and, through it,
    ``ArchiveRegistry.lookup_aliases`` — the package's existing exact-title
    convention; the title this source hands the index already uses spaces, so
    the comparison needs the same fold both sides.

    Measured on the pinned archive (190-question selection, 250 extracted
    spans): where an exact match exists it is the suggest result at rank 1 in
    155/158 cases and never deeper than rank 2 — always inside the ``limit``
    window, so no separate scan depth is needed.
    """
    wanted = span.strip().lower()
    if not wanted:
        return []
    out: list[str] = []
    for p in paths:
        if entry_title_key(p) == wanted:
            out.append(p)
    return out


@register("candidate_source", "title_entity_suggest")
class TitleEntitySuggest:
    """Suggest-index lookup per entity-shaped span from the raw question.

    Where ``title_suggest`` feeds the title index the whole normalized query
    (a prefix nothing matches) and ``alias_title_resolve`` resolves terms the
    redirect table already knows, this source looks up the entities the user
    actually named — casing and quotes included — which is what the
    measurement showed the index wants (T0 0/150, T1 71/150, T2 85/150).
    """

    requires: ClassVar[frozenset[Capability]] = frozenset()

    class Params(BaseModel):
        """Profile-owned knobs (params, not settings).

        ``limit`` — per-span, per-archive suggest limit: the fallback
        emission width when no exact-title match is in the results. 3: the
        span-hitrate probe matched gold
        case-insensitively, so its "all hits @<=5" hides that its rank-1 hits
        are often case variants (``ABRAHAM_LINCOLN`` ahead of
        ``Abraham_Lincoln``); case-sensitively the exact article is at
        rank 1-2 whenever it exists (250-span window probe, 2026-08-16), so
        3 covers the fallback and ranks 4+ only add funnel competition.

        ``max_spans`` — total spans per query; the latency cap (each span is
        one ``suggest`` per archive, ~3 ms warm, serialized per archive).

        ``include_distinctive_terms`` — T2 mode. Off: measured, it wins 2
        more questions on `standard` but loses 4 on `hybrid` (Stage-A funnel
        replay, /tmp proxy 2026-08-16 + full article-recall run) — the
        generic single-word prefix drags popular articles into a funnel the
        dense side already fills.

        ``rank_offset`` — fusion tempering. RRF weights a candidate
        1/(k+rank); this offsets the source's ranks so a title hit *supports*
        an article (adds RRF mass) without letting a title-only nomination
        outvote articles already backed by xapian + dense. Measured on the
        Stage-A funnel replay: offset 0 loses 5 hybrid questions the offset-4
        config keeps (round-robin fusion amplifies every within-archive rank
        step by the number of archives, so a full-weight junk title hit
        displaces a weakly-backed gold by ~10 union positions).
        """

        limit: int = 3
        max_spans: int = 4
        include_distinctive_terms: bool = False
        rank_offset: int = 4

    def __init__(
        self, params: Params | None = None, archives: ArchiveRegistry | None = None
    ) -> None:
        self._params = params or self.Params()
        self._archives = archives

    async def find(self, q: PreparedQuery, scope: Scope, tr: Trace) -> list[Candidate]:
        """Suggest per span on every enabled archive in scope.

        Per archive, spans run in priority order and hits dedupe by path (the
        first occurrence keeps the stronger span's rank), so a weaker span can
        only add articles, never demote a stronger span's hit. A span whose
        results contain its own article (exact title, case-insensitive)
        emits only that article; otherwise it emits its top ``limit``
        results. Emits at most ``limit * max_spans`` candidates per archive.
        """
        if self._archives is None:
            return []

        from vesta.retrieval.impls._scope import archives_for_scope

        archives = await archives_for_scope(self._archives, scope)
        if not archives:
            return []

        spans = extract_spans(
            q.raw,
            max_spans=self._params.max_spans,
            include_distinctive_terms=self._params.include_distinctive_terms,
        )

        async def _suggest_one(archive: Archive) -> list[Candidate]:
            seen: set[str] = set()
            local: list[Candidate] = []
            for span in spans:
                try:
                    paths = await archive.suggest(span, self._params.limit)
                except Exception:
                    continue
                # Exact-title preference: when the span's own article is in
                # the results (rank 1-2, measured), it is the intent and the
                # non-exact tail is dilution — year variants ("Academy
                # Awards" -> Academy_Awards_2017/_2018), sibling articles
                # ("Abraham Lincoln" -> Abraham_Lincoln_Assassination), and
                # case-variant crowds each cost funnel width that a dense-
                # found gold article is competing for (the three measured
                # full losses behind the rank_offset tempering). With no
                # exact match the span names a topic, not an article, and
                # the top-``limit`` results stand.
                emitted = _exact_title_hits(span, paths) or paths
                for p in emitted:
                    if p in seen:
                        continue
                    seen.add(p)
                    local.append(
                        Candidate(
                            zim_id=archive.id,
                            path=p,
                            source="title_entity_suggest",
                            rank=self._params.rank_offset + len(local),
                            score=None,
                        )
                    )
            return local

        results = await asyncio.gather(*(_suggest_one(a) for a in archives), return_exceptions=True)

        out: list[Candidate] = []
        for r in results:
            if isinstance(r, list):
                out.extend(r)
        return out
