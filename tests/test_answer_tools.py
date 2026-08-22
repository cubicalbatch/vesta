"""``format_search_result`` tests — the answer-layer rendering of a
``RetrievalResult`` as the string the agent's ``search`` tool returns.
"""

from __future__ import annotations

import pytest

from vesta.answer.tools import format_search_result
from vesta.retrieval.contracts import ConfidenceSignals, RetrievalResult, ScoredPassage
from vesta.retrieval.trace import Trace
from vesta.zim.types import Passage


def _search_result(zim_id: int = 1, path: str = "Test_Article") -> RetrievalResult:
    p = Passage(
        zim_id=zim_id,
        path=path,
        ordinal=0,
        char_start=0,
        char_end=20,
        breadcrumb="Test Article > Section",
        text="Some passage text.",
        is_lead=True,
    )
    return RetrievalResult(
        passages=(ScoredPassage(passage=p, score=0.9, source_info="rerank"),),
        cards=(),
        trace=Trace(),
        confidence=ConfidenceSignals(top_score=0.9, score_dropoff=0.6, density=0.6, agreement=0.8),
    )


@pytest.mark.parametrize(
    ("zim_id", "labels", "expected_in", "expected_not_in"),
    [
        (7, None, ["archive-7"], []),
        (7, {7: "Wikipedia"}, ["Wikipedia"], ["archive-7"]),
        (7, {3: "Other"}, ["archive-7"], ["Other"]),
    ],
)
def test_format_search_result_archive_labels(
    zim_id: int,
    labels: dict[int, str] | None,
    expected_in: list[str],
    expected_not_in: list[str],
) -> None:
    result = _search_result(zim_id=zim_id)
    text = format_search_result(result, archive_labels=labels)
    for exp in expected_in:
        assert exp in text
    for exp in expected_not_in:
        assert exp not in text
    if labels is None:
        assert format_search_result(result) == text
