"""Conditional Round-0 reformulation — "name the article this fact would live in".

The last rung of Round-0 query shaping. Hybrid search defaults and
``title_entity_suggest`` capture questions whose gold article is
reachable from the question's own words, casing, or embedding. The residual is
the bucket where the title never appears in the question and dense retrieval
also misses it — answering those needs the one thing lexical and dense
retrieval both lack: a model asking *"what article would this fact live in?"*

Two design rules are load-bearing here:

- **Conditional, never always-on.** The call fires only after a
  Round-0 that already visibly failed (``confidence.top_score`` below
  ``answer.reformulate.trigger_score``). The published finding that blind
  pre-retrieval transformation (HyDE, step-back, multi-query) measured worse
  (``retrieval/impls/conversational_rewrite.py`` docstring) is about always-on
  blind transformation; this is a different mechanism — grounded in a visible
  failure. Healthy questions pay exactly zero.

- **Never make things worse.** On gateway exception, empty
  output, or a near-duplicate of the original query, the original result is
  returned untouched. A replacement must strictly beat the original
  ``top_score`` — the same keep-the-better rule ``_maybe_shorten_search``
  applies to tool-driven searches.

The composition root (``api/answer.py``) builds the gateway-backed
:class:`GatewayReformulator` and calls it from the Round-0 path, exactly as it
wires :class:`~vesta.answer.rewriter.GatewayQueryRewriter` today — ``answer/``
may import ``inference/``; ``retrieval/`` stays unaware an LLM exists.

Prompt variants for evaluation: ``exemplified`` is the
purpose-built prompt with worked examples derived from the *shapes* measured in
the residual set (fact buried in a broad topic article, entity named by role,
renamed organisation, brand → generic, plain entity) — never from its specific
questions, to avoid circular evaluation. ``minimal`` is the control
arm: conservative, zero examples.
"""

from __future__ import annotations

import re
import string

from vesta.inference.gateway import ChatMessage, Gateway

#: Exemplified arm. Worked examples, good-and-bad paired on
#: the fact-shaped trap. The framing sentence echoes the bench-tuned query
#: rules in ``api/agent_chat.py``'s SYSTEM_PROMPT — the language the repo has
#: already measured the 4b-class models follow.
REFORMULATE_SYSTEM_PROMPT = """An encyclopedia search failed: the question below was \
searched against an offline Wikipedia archive and the articles it returned do not \
contain the answer. Name the article where the missing fact would live.

This is not rewriting the question. Ask yourself: "what article would this fact \
live in?" — then output that article's NAME as a short keyword query.

Rules:
- Output the article NAME only: the subject the fact is about, 1-4 words, \
capitalized like a title, no quotes, no explanation.
- One query per line, strongest first.
- If the question names a person by role ("who replaced X", "who succeeded X"), \
output that person's own name.
- If it names a company by a former name, output the CURRENT name.
- If it names a product by brand name, output the generic/real name.
- If the fact lives in a broad overview article rather than a dedicated one, \
output the overview article's name.
- Never carry fact-words from the question ("how many", "age", "year", "value", \
"speed") into the query — the search engine matches words, and fact-words bury \
the article name.

Examples:
Q: How old was Michael Faraday when he died?
Michael Faraday

Q: Who became Prime Minister of the United Kingdom after Gordon Brown?
David Cameron

Q: The company formerly known as Andersen Consulting is now called what?
Accenture

Q: What is the active ingredient in the heartburn medicine sold as Zantac?
Ranitidine

Q: What fraction of Earth's atmosphere is argon?
Atmosphere of Earth

Q: Which ship answered the Titanic's distress call and rescued the survivors?
RMS Carpathia

Q: Which country's flag is the only national flag that is not rectangular?
Flag of Nepal

Q: How many strings does a standard balalaika have?
Balalaika

Q: At what voltage did the first transatlantic telegraph cable operate?
Transatlantic telegraph cable"""

#: Minimal arm — the A/B control. Same posture as the conversational
#: rewriter's ``REWRITE_SYSTEM_PROMPT`` (conservative, no examples, "do not
#: expand"), which is the prompt the 679edc2-era belief was formed on. If this
#: arm matches the exemplified one, the examples carry no weight; if both lose,
#: the mechanism is confirmed off on current hardware.
MINIMAL_REFORMULATE_SYSTEM_PROMPT = (
    "You reformulate a question into a search query for an offline encyclopedia "
    "after the first search failed. Output ONLY the reformulated query — one "
    "per line, no preamble, no quotation marks, no explanation. Do not answer "
    "the question. Do not expand the query, and do not add words the user did "
    "not imply."
)

#: A reformulation longer than this is not an article name — it is the question
#: re-phrased (fact-shaped), which the prompt forbids and re-searching would
#: only re-run the original failed AND-match.
_MAX_QUERY_WORDS = 6

#: Token-set Jaccard against the original question at/above this is a
#: near-duplicate re-search: same words, same failure (stagnation).
_DUPLICATE_JACCARD = 0.75

#: Leading decorations small models add: bullets/arrows (possibly stacked,
#: "- → like this"), enumerator numbering ("1." / "2)"), and the "Query:"
#: label — all stripped so the content survives. A bare digit-run title
#: ("Boeing 747") survives; digits are stripped only when enumerator
#: punctuation follows. ``Q:``/``A:`` lines are NOT handled here — they are
#: question echoes and get dropped whole below.
_DECORATION_RE = re.compile(r"^\s*(?:[-•*>→]+\s*|\d+\s*[.):]\s*|query\s*:\s*)+", re.IGNORECASE)

_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "by",
        "for",
        "from",
        "in",
        "into",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "whose",
        "why",
        "how",
        "many",
        "much",
        "did",
        "does",
        "do",
        "has",
        "have",
        "had",
        "not",
        "with",
        "after",
        "before",
        "between",
        "over",
        "under",
        "about",
    ]
)


def _normalize(text: str) -> set[str]:
    """Lowercased content-word tokens (stopwords dropped) for dup comparison."""
    tokens = set()
    for w in text.lower().translate(str.maketrans("", "", string.punctuation)).split():
        if w and w not in _STOPWORDS:
            tokens.add(w)
    return tokens


def _is_duplicate(candidate: str, original: str) -> bool:
    """True when ``candidate`` re-searches the same words as ``original``."""
    a, b = _normalize(candidate), _normalize(original)
    if not a or not b:
        return False
    return len(a & b) / len(a | b) >= _DUPLICATE_JACCARD


def parse_reformulations(text: str, *, limit: int, original: str) -> list[str]:
    """Parse the model's reply into entity-shaped queries, strongest first.

    Tolerates the decorations small models add (``1.``, ``-``, ``→``, quotes,
    ``Query:``) and drops everything that cannot be an article name: blank
    lines, the prompt's own ``Q:`` echoes, parenthesized explanations, lines
    longer than :data:`_MAX_QUERY_WORDS` words, duplicates of each other, and
    near-duplicates of the original question (stagnation). Caps at ``limit``.
    """
    queries: list[str] = []
    seen: set[frozenset[str]] = set()
    for raw_line in text.splitlines():
        line = _DECORATION_RE.sub("", raw_line).strip()
        line = line.strip('"').strip("'").strip()
        if not line or line.startswith("("):
            continue
        low = line.lower()
        if low.startswith(("q:", "query:", "a:", "answer:", "bad:", "good:", "not ")):
            continue
        words = line.split()
        if not 1 <= len(words) <= _MAX_QUERY_WORDS:
            continue
        if _is_duplicate(line, original):
            continue
        key = frozenset(_normalize(line))
        if key in seen:  # same tokens reordered adds nothing to a word matcher
            continue
        seen.add(key)
        queries.append(line)
        if len(queries) >= max(1, limit):
            break
    return queries


class GatewayReformulator:
    """One ``gateway.chat_once`` call that names the article to re-search with.

    Same posture as :class:`~vesta.answer.rewriter.GatewayQueryRewriter`:
    ``enable_thinking=False`` (a hidden reasoning chain would burn the whole
    64-token budget before emitting the query), temperature 0.0,
    and a token budget sized for 1-2 article names, deliberately too small for
    an essay. Raises only if the gateway call itself fails; the caller's
    never-worse contract does the rest.
    """

    def __init__(
        self,
        gateway: Gateway,
        *,
        model: str,
        max_tokens: int = 64,
        temperature: float = 0.0,
        enable_thinking: bool = False,
        prompt: str = REFORMULATE_SYSTEM_PROMPT,
    ) -> None:
        self._gateway = gateway
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._enable_thinking = enable_thinking
        self._prompt = prompt

    @property
    def prompt(self) -> str:
        """The active system prompt (exposed for the A/B verification tests)."""
        return self._prompt

    async def reformulate(self, question: str, *, limit: int = 1) -> list[str]:
        """Return 0..``limit`` article-name queries for a failed Round 0.

        An empty list means "nothing usable" (empty, unparsable, or stagnant
        output) and the caller keeps the original result untouched.
        """
        messages: list[ChatMessage] = [
            ChatMessage(role="system", content=self._prompt),
            ChatMessage(role="user", content=question),
        ]
        result = await self._gateway.chat_once(
            messages,
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            enable_thinking=self._enable_thinking,
        )
        return parse_reformulations(result.text, limit=limit, original=question)


__all__ = [
    "MINIMAL_REFORMULATE_SYSTEM_PROMPT",
    "REFORMULATE_SYSTEM_PROMPT",
    "GatewayReformulator",
    "parse_reformulations",
]
