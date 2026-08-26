"""Normalize query preparer — trivially cleans the raw query string.

Registered as ``query_preparer`` ``normalize``. Always the first preparer in every
profile: lowercases the raw text, strips whitespace, and sets ``is_keyword_query``
from a simple heuristic. This is the bootstrap step that turns a raw user string
into a ``PreparedQuery`` the rest of the pipeline consumes.

Speed: trivial (string ops only, no I/O).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel

from vesta.config.capabilities import Capability
from vesta.retrieval.contracts import PreparedQuery
from vesta.retrieval.registry import register
from vesta.zim.query import DEFAULT_STOPWORDS, INERT_TOKENS, looks_like_question

if TYPE_CHECKING:
    from vesta.retrieval.trace import Trace


def _is_keyword_query(text: str) -> bool:
    """True when ``text`` looks like a lookup, not a question.

    A query ending in ``?`` or starting with an interrogative is a question;
    everything else is a keyword query (the "bare search term" mode).
    """
    return not looks_like_question(text)


@register("query_preparer", "normalize")
class Normalize:
    """Trivial cleanup preparer — always first in the chain.

    Also strips stopwords/interrogatives from ``terms`` (not from ``text``,
    which downstream keyword-vs-question classification and display still
    need whole). Every candidate source reads ``q.terms``, and libzim's parser
    makes every term mandatory — without this, a natural-language
    question like "what is photosynthesis" ANDs in "what" and "is", either
    matching nothing or matching unrelated pages that happen to contain those
    common words, which starves the real answer of a look-in. This mirrors
    ``zim.query.QueryPreparer``'s stopword rung, but runs unconditionally
    instead of only as a last-resort fallback when a source returns literally
    zero results — a coincidental noise match was silently defeating that
    fallback's trigger condition. It also drops libzim-inert boolean/operator
    tokens (``vesta.zim.query.INERT_TOKENS``), matching what
    ``zim.query.normalize_terms`` does on the ladder path, so both term
    pipelines agree on the primary path too.
    """

    requires: ClassVar[frozenset[Capability]] = frozenset()

    class Params(BaseModel):
        strip_stopwords: bool = True
        stopwords: str = ",".join(DEFAULT_STOPWORDS)

    def __init__(self, params: Params | None = None) -> None:
        self._params = params or self.Params()
        self._stopwords = frozenset(
            w.strip() for w in self._params.stopwords.split(",") if w.strip()
        )

    async def prepare(self, q: PreparedQuery, tr: Trace) -> PreparedQuery:
        """Normalize: lowercase the raw text, strip whitespace, detect keyword vs
        question form. Returns a new ``PreparedQuery`` — the input is the raw
        pipeline bootstrap, and this is the first transform."""
        text = q.raw.strip().lower()
        # Simple word-level term extraction for the bootstrap.
        terms = tuple(
            w for w in text.replace("?", " ").replace('"', " ").replace("'", " ").split() if w
        )
        if self._params.strip_stopwords:
            stripped = tuple(t for t in terms if t not in self._stopwords)
            # Only adopt the stripped form if it didn't remove everything — a
            # query that is entirely stopwords (rare) still needs something to
            # search with.
            if stripped:
                terms = stripped
        # Drop libzim-inert boolean/operator tokens ("not", "near", "xor",
        # standalone "+"/"-"/"*") — the same set ``zim.query.normalize_terms``
        # drops on the ladder path. Without this they become literal AND-ed
        # terms on the primary path ("hotels near the grand canyon" requires a
        # literal "near"), while the ladder only rescued that on a zero-result
        # fallback. Same empty-guard as above: never strip down to nothing.
        inert_stripped = tuple(t for t in terms if t not in INERT_TOKENS)
        if inert_stripped:
            terms = inert_stripped
        result = PreparedQuery(
            raw=q.raw,
            terms=terms,
            text=text,
            aliases=q.aliases,
            is_keyword_query=_is_keyword_query(text),
            rung=q.rung,
            history=q.history,
        )
        return result
