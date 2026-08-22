"""Context assembly strategies — the "context
strategy" half of this project's core, and the part most worth making plural.

Four registered ``context_assembler`` implementations, each independent of how
passages were *scored* (registered behind a fixed contract,
no caller branches on which is active):

* ``topk_budget`` — top-N to a token budget.
* ``section_window`` — expand winning passages to their section neighbours.
* ``lead_boost`` — boost the lead passage of a strongly-matching article.
* ``diverse`` — MMR-style, caps per-article and per-archive concentration.

Importing these modules here registers them; ``retrieval/__init__.py`` imports
this package before profiles are loaded.
"""

from __future__ import annotations

from vesta.retrieval.assemblers import diverse, lead_boost, section_window, topk_budget

__all__ = ["diverse", "lead_boost", "section_window", "topk_budget"]
