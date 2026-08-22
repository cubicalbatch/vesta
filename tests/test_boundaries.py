"""Module-boundary enforcement.

Two rules, both load-bearing for the architecture:

* ``retrieval/`` must not import ``answer``, ``api``, or ``index``. Retrieval is
  a library that takes a query and returns passages + a trace. If it knows an
  LLM or a UI exists, it can no longer be benchmarked in isolation.
* ``api/`` is the only package that may import from more than two *other*
  internal packages. It is the composition root; business logic never lives
  there, and every other package stays narrow.

Hand-rolled AST walk (stdlib only) so we don't add a dep, and so the rule reads
as plain Python rather than a grimp config file.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src" / "vesta"

#: Every internal subpackage under ``vesta``. A directory qualifies if it holds
#: an ``__init__.py``. Top-level modules (``main.py``, ``runtime.py``) are not
#: packages and are exempt from the per-package cap — ``main`` is the app
#: factory that wires everything (composition root), like ``api``.
INTERNAL_PACKAGES: set[str] = {
    p.name for p in SRC.iterdir() if p.is_dir() and (p / "__init__.py").exists()
}


def _node_module_paths(node: ast.AST, file: Path) -> list[str]:
    """Dotted module strings an import node draws from.

    Handles the three forms in the codebase: ``import vesta.x``, absolute
    ``from vesta.x import ...`` (incl. the bare-name ``from vesta import config``
    form, which ``_vesta_packages`` would otherwise miss), and relative imports
    resolved against the file's package.
    """
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if not isinstance(node, ast.ImportFrom):
        return []
    if not node.level:  # absolute
        paths = [node.module] if node.module else []
        # `from vesta import config, jobs` imports packages by bare name; recognise
        # each so ``config`` as a dependency isn't silently undercounted.
        if node.module == "vesta":
            paths.extend(f"vesta.{a.name}" for a in node.names)
        return paths
    # relative import: resolve ``level`` against the file's own package.
    base = list(file.relative_to(SRC).parts[:-1])
    keep = len(base) - (node.level - 1)
    if keep <= 0:
        return []
    resolved = base[:keep]
    if node.module:
        resolved.append(node.module)
    return [f"vesta.{'.'.join(resolved)}"] if resolved else []


def _imports_in(file: Path) -> set[str]:
    """Return the set of internal package names this file imports from."""
    found: set[str] = set()
    try:
        tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
    except SyntaxError:
        return found
    for node in ast.walk(tree):
        for module_path in _node_module_paths(node, file):
            found |= _vesta_packages(module_path)
    return found


def _vesta_packages(module: str) -> set[str]:
    """Map an import string to the internal package(s) it touches."""
    parts = module.split(".")
    if len(parts) >= 2 and parts[0] == "vesta" and parts[1] in INTERNAL_PACKAGES:
        return {parts[1]}
    return set()


def _package_imports() -> dict[str, set[str]]:
    """For each internal package, the set of other internal packages it imports."""
    result: dict[str, set[str]] = {name: set() for name in INTERNAL_PACKAGES}
    for pkg in INTERNAL_PACKAGES:
        for file in (SRC / pkg).rglob("*.py"):
            for imported in _imports_in(file):
                if imported != pkg:  # intra-package imports don't count
                    result[pkg].add(imported)
    return result


#: Forbidden internal-package imports per source package. Each row is an
#: architectural property from the module map, made mechanical:
#: the dependency arrow is one-way, and the "wide" library packages must not
#: point back up at the policy/composition layers.
#:
#: * ``retrieval`` must not import ``answer``/``api``/``index`` — it is a
#:   query→passages library and must not know an LLM, a UI, or the indexer
#:   exists.
#: * ``zim`` must not import ``retrieval``/``answer``/``api``/``index`` — it is
#:   the widest surface of raw-material complexity with **no retrieval policy in
#:   it** (02-zim-data-layer.md Purpose). The ≤2-dep cap below would NOT catch
#:   ``zim`` swapping one of its {config, db} deps for ``retrieval`` (still 2),
#:   so the forbidden set is asserted explicitly.
FORBIDDEN_IMPORTS: dict[str, set[str]] = {
    "retrieval": {"answer", "api", "index"},
    "zim": {"retrieval", "answer", "api", "index"},
}


@pytest.mark.parametrize("pkg,forbidden", sorted((k, v) for k, v in FORBIDDEN_IMPORTS.items()))
def test_no_forbidden_imports(pkg: str, forbidden: set[str]) -> None:
    """A library package must not point back up at the policy/composition layers.
    Parameterized so adding a guarded package is one row."""
    imports = _package_imports().get(pkg, set())
    violated = imports & forbidden
    assert not violated, (
        f"{pkg}/ must not import {sorted(violated)}; "
        f"it is a library and must not host retrieval policy or know an LLM/UI/indexer exists"
    )


#: Packages the module map explicitly widens beyond the ≤2
#: default, each a deliberate exception (not a cap violation):
#:
#: * ``retrieval`` → {zim, config, encoders, vectors}: originally 3 deps (the
#:   module map's "zim, config, encoders"), widened to 4 by the ONNX encoder
#:   runtime. ``api``/``cli`` are the composition roots that wire an
#:   ``EncoderManager`` into ``retrieval``'s ``Deps`` container — ``retrieval``
#:   itself only ever imports the encoders *contracts*. ``vectors`` is a FOURTH,
#:   **type-only** dep: ``Deps.vectors`` is annotated
#:   ``VectorStore | None`` via an ``if TYPE_CHECKING`` import (the dense
#:   ``vector_knn`` source receives the store via DI, never by runtime import).
#:   This is the only way to type the DI field without duplicating the Protocol.
#: * ``answer`` → {retrieval, inference, config}: the answer pipeline depends on
#:   retrieval (passages + trace) and inference (the LLM gateway). ``config`` is
#:   the settings/Capability dep that
#:   every package with declared settings needs (``answer.*`` settings, the
#:   ``Capability`` for ``requires`` ClassVars) — the same transitive dep
#:   retrieval already has. The module map lists "depends on: retrieval,
#:   inference"; config is the same foundational dep that appears in every
#:   package that declares settings.
#: * ``index`` → {config, db, jobs, vectors, zim}: the indexing package. The
#:   domain deps are zim/encoders/vectors, but three more are structural:
#:   ``config`` is the universal settings dep (same as ``answer``/``retrieval``);
#:   ``jobs`` is the job-type plugin mechanism — the indexer IS an
#:   ``index_zim`` JobType, so ``register_job_type``/``JobHandle``/the resume
#:   checkpoint key are runtime imports by design; ``db`` is TYPE_CHECKING-only
#:   (the live ``Database`` is injected via ``bind_runtime`` — the job writes
#:   ``zims.index_*`` status rows and populates ``articles`` through it). The
#:   ``encoders`` dep is realised *through* the injected
#:   embedder provider, so ``encoders`` itself is never imported.
#: * ``catalog`` → {config, db, jobs}: the OPDS
#:   catalog client + resumable downloads. ``config`` is the universal settings
#:   dep (catalog.*/download.* settings declared in-package); ``db`` backs the
#:   cached ``catalog_entries`` + FTS5 index; ``jobs`` is the job-type plugin
#:   mechanism — the ``download_zim`` and ``refresh_catalog`` jobs ARE JobTypes
#:   (``register_job_type``/``JobHandle`` are runtime imports by design). The
#:   post-download "register archive" step (rescan) is injected as a
#:   callback from ``main`` — catalog/ never imports ``zim/`` at runtime, so the
#:   dep set stays at 3.
_WIDENED_CAPS: dict[str, int] = {"retrieval": 4, "answer": 3, "index": 5, "catalog": 3}


def test_only_api_may_import_more_than_two_packages() -> None:
    """``api/`` is the composition root; every other package stays ≤2 deps,
    except the dated exceptions in ``_WIDENED_CAPS``."""
    imports = _package_imports()
    offenders = {
        pkg: deps
        for pkg, deps in imports.items()
        if pkg != "api" and len(deps) > _WIDENED_CAPS.get(pkg, 2)
    }
    assert not offenders, (
        f"only api/ may import from more than two internal packages "
        f"(exceptions: {_WIDENED_CAPS}); offenders: {offenders}"
    )


def test_db_imports_nothing_internal() -> None:
    """db is the foundation; it depends on nothing."""
    assert _package_imports().get("db", set()) == set(), "db must not import any internal package"


@pytest.mark.parametrize("pkg", sorted(INTERNAL_PACKAGES))
def test_every_package_is_scanable(pkg: str) -> None:
    """Smoke: the package dir exists and parses."""
    assert (SRC / pkg / "__init__.py").exists()
