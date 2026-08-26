"""Tests for pipeline orchestration and degradation recording.

Ensures that when any component (preparer, candidate source, fuser, passage builder,
or passage scorer) fails to instantiate or raises an exception during execution,
a degradation record is appended to the trace so downstream comparisons and
evaluations do not consider the run clean.
"""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

import pytest

from vesta.config.capabilities import Capability
from vesta.retrieval.contracts import (
    Candidate,
    PreparedQuery,
    RetrievalResult,
    Scope,
    ScoredPassage,
)
from vesta.retrieval.pipeline import Deps, run_pipeline
from vesta.retrieval.profiles import ProfileComponent, RetrievalProfile
from vesta.retrieval.registry import COMPONENTS, ComponentParams
from vesta.zim.types import EntryFlags, ExtractedArticle, Passage

# ── Fakes & Test Components ──────────────────────────────────────────────────


@dataclass
class _FakeArchive:
    id: int = 1
    uuid: str = "fake-uuid"
    title: str = "Fake Archive"
    language: str = "en"
    has_fulltext_index: bool = True
    article_count: int = 10

    async def search(self, terms: Any, limit: int) -> list[str]:
        return ["Article_1", "Article_2"]

    async def suggest(self, prefix: str, limit: int) -> list[str]:
        return ["Article_1"]

    async def read(self, path: str) -> None:
        raise NotImplementedError

    async def extract(self, path: str) -> Any:
        return ExtractedArticle(
            path=path,
            title="Fake Article",
            text="This is a test article with content for retrieval testing.",
            sections=(),
            flags=EntryFlags.NONE,
        )

    async def main_path(self) -> str:
        return "Article_1"


class _FakeRegistry:
    def __init__(self, archive: _FakeArchive | None = None) -> None:
        self._archive = archive or _FakeArchive()

    def get(self, zim_id: int) -> _FakeArchive:
        return self._archive

    def enabled(self, scope: Any = None) -> list[_FakeArchive]:
        return [self._archive]

    def has_any_fulltext(self) -> bool:
        return self._archive.has_fulltext_index

    async def lookup_aliases(self, terms: Any, *, max_aliases: int) -> list[str]:
        return []

    async def ids_for_labels(self, labels: Any) -> frozenset[int]:
        return frozenset()


class _GoodSource:
    requires = frozenset()

    class Params(ComponentParams):
        pass

    def __init__(self, params: Any = None, archives: Any = None) -> None:
        pass

    async def find(self, q: PreparedQuery, scope: Scope, tr: Any) -> list[Candidate]:
        return [
            Candidate(
                zim_id=1,
                path="Article_1",
                source="test_good_source",
                rank=0,
                score=1.0,
            )
        ]


class _CrashingSourceInit:
    requires = frozenset()

    class Params(ComponentParams):
        pass

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("failed to initialize source")

    async def find(self, q: PreparedQuery, scope: Scope, tr: Any) -> list[Candidate]:
        return []


class _CrashingSourceFind:
    requires = frozenset()

    class Params(ComponentParams):
        pass

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def find(self, q: PreparedQuery, scope: Scope, tr: Any) -> list[Candidate]:
        raise RuntimeError("search exception in find")


class _CrashingPreparerInit:
    requires = frozenset()

    class Params(ComponentParams):
        pass

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("failed to initialize preparer")

    async def prepare(self, q: PreparedQuery, tr: Any) -> PreparedQuery:
        return q


class _CrashingPreparerPrepare:
    requires = frozenset()

    class Params(ComponentParams):
        pass

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def prepare(self, q: PreparedQuery, tr: Any) -> PreparedQuery:
        raise RuntimeError("rewrite exception in prepare")


class _CrashingFuserInit:
    requires = frozenset()

    class Params(ComponentParams):
        pass

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("failed to initialize fuser")

    def fuse(self, groups: Any, tr: Any) -> list[Candidate]:
        return []


class _CrashingFuserFuse:
    requires = frozenset()

    class Params(ComponentParams):
        pass

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def fuse(self, groups: Any, tr: Any) -> list[Candidate]:
        raise ZeroDivisionError("k cannot be zero")


class _CrashingBuilderInit:
    requires = frozenset()

    class Params(ComponentParams):
        pass

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("failed to initialize builder")

    async def build(self, candidates: Any, q: PreparedQuery, tr: Any) -> list[Passage]:
        return []


class _CrashingBuilderBuild:
    requires = frozenset()

    class Params(ComponentParams):
        pass

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def build(self, candidates: Any, q: PreparedQuery, tr: Any) -> list[Passage]:
        raise RuntimeError("builder exception in build")


class _CrashingScorerInit:
    requires = frozenset()

    class Params(ComponentParams):
        pass

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("failed to initialize scorer")

    async def score(
        self, passages: list[ScoredPassage], q: PreparedQuery, tr: Any
    ) -> list[ScoredPassage]:
        return passages


class _CrashingScorerScore:
    requires = frozenset()

    class Params(ComponentParams):
        pass

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def score(
        self, passages: list[ScoredPassage], q: PreparedQuery, tr: Any
    ) -> list[ScoredPassage]:
        raise RuntimeError("scorer forward pass exception")


@pytest.fixture(autouse=True)
def setup_test_components() -> Generator[None]:
    original = dict(COMPONENTS)
    COMPONENTS[("candidate_source", "test_good_source")] = _GoodSource
    COMPONENTS[("candidate_source", "test_crashing_source_init")] = _CrashingSourceInit
    COMPONENTS[("candidate_source", "test_crashing_source_find")] = _CrashingSourceFind
    COMPONENTS[("query_preparer", "test_crashing_preparer_init")] = _CrashingPreparerInit
    COMPONENTS[("query_preparer", "test_crashing_preparer_prepare")] = _CrashingPreparerPrepare
    COMPONENTS[("fuser", "test_crashing_fuser_init")] = _CrashingFuserInit
    COMPONENTS[("fuser", "test_crashing_fuser_fuse")] = _CrashingFuserFuse
    COMPONENTS[("passage_builder", "test_crashing_builder_init")] = _CrashingBuilderInit
    COMPONENTS[("passage_builder", "test_crashing_builder_build")] = _CrashingBuilderBuild
    COMPONENTS[("passage_scorer", "test_crashing_scorer_init")] = _CrashingScorerInit
    COMPONENTS[("passage_scorer", "test_crashing_scorer_score")] = _CrashingScorerScore
    try:
        yield
    finally:
        COMPONENTS.clear()
        COMPONENTS.update(original)


def _make_profile(**overrides: Any) -> RetrievalProfile:
    return RetrievalProfile(
        name="test",
        description="Test profile",
        hash="test-hash",
        preparers=overrides.get("preparers", (ProfileComponent(impl="normalize"),)),
        sources=overrides.get("sources", (ProfileComponent(impl="test_good_source"),)),
        fusion=overrides.get(
            "fusion", ProfileComponent(impl="rrf", params={"k": 20, "across_archives": "union"})
        ),
        passages=overrides.get(
            "passages",
            ProfileComponent(
                impl="candidate_articles", params={"max_articles": 5, "max_passages": 20}
            ),
        ),
        scorers=overrides.get("scorers", (ProfileComponent(impl="lexical_overlap"),)),
        assembler=overrides.get(
            "assembler",
            ProfileComponent(
                impl="topk_budget",
                params={"budget_tokens": 1000, "max_per_article": 2, "dedup": "none"},
            ),
        ),
    )


def _deps() -> Deps:
    return Deps(
        archives=_FakeRegistry(),  # type: ignore[arg-type]
        capabilities=frozenset({Capability.ZIM_FULLTEXT}),
    )


# ── Tests ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_preparer_init_failure_records_degradation() -> None:
    profile = _make_profile(
        preparers=(
            ProfileComponent(impl="test_crashing_preparer_init"),
            ProfileComponent(impl="normalize"),
        )
    )
    result = await run_pipeline(profile, "test query", Scope(), _deps())
    assert isinstance(result, RetrievalResult)
    degs = result.trace.to_dict()["degradations"]
    assert any(
        d["component"] == "preparer/test_crashing_preparer_init"
        and d["missing"] == "runtime_error"
        and "failed to initialize preparer" in d["reason"]
        for d in degs
    )


@pytest.mark.asyncio
async def test_preparer_execution_failure_records_degradation() -> None:
    profile = _make_profile(
        preparers=(
            ProfileComponent(impl="test_crashing_preparer_prepare"),
            ProfileComponent(impl="normalize"),
        )
    )
    result = await run_pipeline(profile, "test query", Scope(), _deps())
    assert isinstance(result, RetrievalResult)
    degs = result.trace.to_dict()["degradations"]
    assert any(
        d["component"] == "preparer/test_crashing_preparer_prepare"
        and d["missing"] == "runtime_error"
        and "rewrite exception in prepare" in d["reason"]
        for d in degs
    )


@pytest.mark.asyncio
async def test_preparer_not_found_records_degradation() -> None:
    profile = _make_profile(
        preparers=(
            ProfileComponent(impl="nonexistent_preparer"),
            ProfileComponent(impl="normalize"),
        )
    )
    result = await run_pipeline(profile, "test query", Scope(), _deps())
    assert isinstance(result, RetrievalResult)
    degs = result.trace.to_dict()["degradations"]
    assert any(
        d["component"] == "preparer/nonexistent_preparer" and d["missing"] == "not_found"
        for d in degs
    )


@pytest.mark.asyncio
async def test_candidate_source_init_failure_records_degradation() -> None:
    profile = _make_profile(
        sources=(
            ProfileComponent(impl="test_crashing_source_init"),
            ProfileComponent(impl="test_good_source"),
        )
    )
    result = await run_pipeline(profile, "test query", Scope(), _deps())
    assert isinstance(result, RetrievalResult)
    degs = result.trace.to_dict()["degradations"]
    assert any(
        d["component"] == "candidate_source/test_crashing_source_init"
        and d["missing"] == "runtime_error"
        and "failed to initialize source" in d["reason"]
        for d in degs
    )


@pytest.mark.asyncio
async def test_candidate_source_execution_failure_records_degradation() -> None:
    profile = _make_profile(
        sources=(
            ProfileComponent(impl="test_crashing_source_find"),
            ProfileComponent(impl="test_good_source"),
        )
    )
    result = await run_pipeline(profile, "test query", Scope(), _deps())
    assert isinstance(result, RetrievalResult)
    degs = result.trace.to_dict()["degradations"]
    assert any(
        d["component"] == "candidate_source/test_crashing_source_find"
        and d["missing"] == "runtime_error"
        and "search exception in find" in d["reason"]
        for d in degs
    )


@pytest.mark.asyncio
async def test_candidate_source_not_found_records_degradation() -> None:
    profile = _make_profile(
        sources=(
            ProfileComponent(impl="nonexistent_source"),
            ProfileComponent(impl="test_good_source"),
        )
    )
    result = await run_pipeline(profile, "test query", Scope(), _deps())
    assert isinstance(result, RetrievalResult)
    degs = result.trace.to_dict()["degradations"]
    assert any(
        d["component"] == "candidate_source/nonexistent_source" and d["missing"] == "not_found"
        for d in degs
    )


@pytest.mark.asyncio
async def test_fuser_init_failure_falls_back_and_records_degradation() -> None:
    profile = _make_profile(fusion=ProfileComponent(impl="test_crashing_fuser_init"))
    result = await run_pipeline(profile, "test query", Scope(), _deps())
    assert isinstance(result, RetrievalResult)
    degs = result.trace.to_dict()["degradations"]
    assert any(
        d["component"] == "fuser/test_crashing_fuser_init"
        and d["missing"] == "runtime_error"
        and "failed to initialize fuser" in d["reason"]
        for d in degs
    )


@pytest.mark.asyncio
async def test_fuser_execution_failure_falls_back_and_records_degradation() -> None:
    """AUDIT_0824 R3: fuser exception in fuse() must record degradation and fall back to flat union."""
    profile = _make_profile(fusion=ProfileComponent(impl="test_crashing_fuser_fuse"))
    result = await run_pipeline(profile, "test query", Scope(), _deps())
    assert isinstance(result, RetrievalResult)
    degs = result.trace.to_dict()["degradations"]
    assert any(
        d["component"] == "fuser/test_crashing_fuser_fuse"
        and d["missing"] == "runtime_error"
        and "k cannot be zero" in d["reason"]
        for d in degs
    )
    # Ensure stage output recorded error as well
    stages = result.trace.to_dict()["stages"]
    fuser_stage = next(s for s in stages if s["name"] == "fuser")
    assert "error" in fuser_stage["outputs"]
    assert "k cannot be zero" in fuser_stage["outputs"]["error"]


@pytest.mark.asyncio
async def test_passage_builder_init_failure_records_degradation() -> None:
    profile = _make_profile(passages=ProfileComponent(impl="test_crashing_builder_init"))
    result = await run_pipeline(profile, "test query", Scope(), _deps())
    assert isinstance(result, RetrievalResult)
    degs = result.trace.to_dict()["degradations"]
    assert any(
        d["component"] == "passage_builder/test_crashing_builder_init"
        and d["missing"] == "runtime_error"
        and "failed to initialize builder" in d["reason"]
        for d in degs
    )


@pytest.mark.asyncio
async def test_passage_builder_execution_failure_records_degradation() -> None:
    profile = _make_profile(passages=ProfileComponent(impl="test_crashing_builder_build"))
    result = await run_pipeline(profile, "test query", Scope(), _deps())
    assert isinstance(result, RetrievalResult)
    degs = result.trace.to_dict()["degradations"]
    assert any(
        d["component"] == "passage_builder/test_crashing_builder_build"
        and d["missing"] == "runtime_error"
        and "builder exception in build" in d["reason"]
        for d in degs
    )


@pytest.mark.asyncio
async def test_passage_scorer_init_failure_records_degradation() -> None:
    profile = _make_profile(scorers=(ProfileComponent(impl="test_crashing_scorer_init"),))
    result = await run_pipeline(profile, "test query", Scope(), _deps())
    assert isinstance(result, RetrievalResult)
    degs = result.trace.to_dict()["degradations"]
    assert any(
        d["component"] == "passage_scorer/test_crashing_scorer_init"
        and d["missing"] == "runtime_error"
        and "failed to initialize scorer" in d["reason"]
        for d in degs
    )


@pytest.mark.asyncio
async def test_passage_scorer_execution_failure_records_degradation() -> None:
    profile = _make_profile(scorers=(ProfileComponent(impl="test_crashing_scorer_score"),))
    result = await run_pipeline(profile, "test query", Scope(), _deps())
    assert isinstance(result, RetrievalResult)
    degs = result.trace.to_dict()["degradations"]
    assert any(
        d["component"] == "passage_scorer/test_crashing_scorer_score"
        and d["missing"] == "runtime_error"
        and "scorer forward pass exception" in d["reason"]
        for d in degs
    )
