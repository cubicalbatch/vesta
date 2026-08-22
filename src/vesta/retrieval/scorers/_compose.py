"""Shared helper: compose a passage's embedding/scoring text.

``Passage.text`` and ``Passage.breadcrumb`` are kept apart so
``char_start``/``char_end`` offsets stay honest (``zim/types.py`` Passage
docstring); the caller composes them at embedding/scoring time. Every Stage B
scorer needs this exact composition, so it lives once, here.
"""

from __future__ import annotations

from vesta.zim.types import Passage


def passage_text(passage: Passage) -> str:
    """``"Article > Section  passage text"`` — matches the models' `title +
    passage` training distribution."""
    if passage.breadcrumb:
        return f"{passage.breadcrumb}  {passage.text}"
    return passage.text


__all__ = ["passage_text"]
