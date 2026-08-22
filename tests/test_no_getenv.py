"""No environment reads outside ``config/``.

The settings table is authoritative; env is first-boot defaults only, and the
*only* place the environment is read is the resolver. We flag every form:
``os.getenv`` calls and any ``os.environ`` access (subscript, ``.get``, or
``dict(os.environ)``).
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "vesta"


def _env_reads(file: Path) -> Iterator[str]:
    try:
        tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
    except SyntaxError:
        return
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "getenv"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "os"
        ):
            # os.getenv(...)
            yield "os.getenv"
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "os"
            and node.attr == "environ"
        ):
            # os.environ (any subsequent use: subscript, .get, dict(...), ...)
            yield "os.environ"


def _collect() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for file in SRC.rglob("*.py"):
        hits = list(_env_reads(file))
        if hits:
            found[str(file.relative_to(SRC))] = hits
    return found


def test_no_env_reads_outside_config() -> None:
    offenders = {path: hits for path, hits in _collect().items() if not path.startswith("config")}
    assert not offenders, f"os.getenv / os.environ may only appear in config/; found: {offenders}"


def test_config_is_the_env_source() -> None:
    """Sanity: the resolver reads the environment, exactly once-ish."""
    config_hits = [
        h for path, hits in _collect().items() if path.startswith("config") for h in hits
    ]
    assert config_hits, "config/resolution.py should read os.environ as the one env source"
