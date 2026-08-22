"""Retrieval component implementations.

Each module registers one or more implementations via ``@register(kind, name)``.
Importing modules here is what makes them visible to the registry — there is no
plugin discovery magic. Adding a new component means creating
a new module and adding ``from vesta.retrieval.impls import new_module`` below.

Rules:
* An implementation never imports a sibling implementation. Shared logic moves
  down into a plain function.
* Implementations receive dependencies through constructor injection (``Deps``
  container), never by importing singletons.
"""

from __future__ import annotations

from vesta.retrieval.impls import (
    alias_expand,
    alias_title_resolve,
    candidate_articles,
    conversational_rewrite,
    lexical_overlap,
    normalize,
    rrf,
    title_entity_suggest,
    title_suggest,
    vector_knn,
    xapian_fts,
)

__all__ = [
    "alias_expand",
    "alias_title_resolve",
    "candidate_articles",
    "conversational_rewrite",
    "lexical_overlap",
    "normalize",
    "rrf",
    "title_entity_suggest",
    "title_suggest",
    "vector_knn",
    "xapian_fts",
]
