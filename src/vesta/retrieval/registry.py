"""Component registry — every swappable pipeline behaviour.

Three rules, all load-bearing:

1. **Params are a Pydantic model attached to the class.** This is what lets the
   dev console render a form for a component nobody wrote UI code for.
2. **``requires`` declares capabilities, not imports.** The pipeline checks this
   against the live capability set and drops unmet components, recording the drop
   in the trace. No import-time side effects.
3. **Registration by explicit import, no discovery magic.** Importing the impl
   modules in the retrieval package ``__init__`` registers them. No entry-point
   scanning, no metaclass tracking, no ``pkg_resources``.

An implementation carries:

* ``requires: ClassVar[frozenset[Capability]]`` — what it needs to function.
* ``Params`` — a Pydantic ``BaseModel`` subclass with typed, defaulted fields.
  Extracted via introspection and served by ``GET /api/retrieval/components`` so
  the profile editor renders per-component forms generically.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from vesta.config.capabilities import Capability

#: All registered components. Key is ``(kind, name)`` — e.g. ``("candidate_source",
#: "xapian_fts")``. The value is the implementation *class* (not an instance).
COMPONENTS: dict[tuple[str, str], type] = {}


def register(kind: str, name: str) -> Any:
    """Decorator: register an implementation. ``kind`` is the contract role,
    ``name`` the impl id. The decorator returns the class unchanged.

    Usage::

        @register("candidate_source", "xapian_fts")
        class XapianFTS:
            requires = frozenset({Capability.ZIM_FULLTEXT})
            ...
    """

    def decorator(cls: Any) -> Any:
        COMPONENTS[(kind, name)] = cls
        return cls

    return decorator


def resolve(kind: str, name: str) -> type | None:
    """Look up a component by kind and name. Returns ``None`` if not found."""
    return COMPONENTS.get((kind, name))


def registered(kind: str) -> list[str]:
    """All registered impl names for ``kind``, in registration order."""
    return [name for (k, name) in sorted(COMPONENTS, key=lambda kn: (kn[0], kn[1])) if k == kind]


def _get_params_schema(cls: type) -> type[BaseModel] | None:
    """Extract the ``Params`` Pydantic model from an implementation class.

    Returns ``None`` if the class has no ``Params`` attribute or it is not a
    ``BaseModel`` subclass (allowed — a component without params is valid).
    """
    params = getattr(cls, "Params", None)
    if params is not None and isinstance(params, type) and issubclass(params, BaseModel):
        return params
    return None


def _get_requires(cls: type) -> frozenset[Capability]:
    """Extract the ``requires`` capability set from an implementation class."""
    req = getattr(cls, "requires", None)
    if isinstance(req, frozenset):
        return req
    return frozenset()


def component_schemas() -> dict[str, list[dict[str, Any]]]:
    """Registry introspection for ``GET /api/retrieval/components``.

    Returns::

        {
            "preparers": [{"name": "normalize", "params_schema": {...}, "requires": [...], ...}],
            ...
        }
    """
    kind_order = ("preparers", "sources", "fusion", "passages", "scorers", "assemblers")
    kinds_map = {
        "preparers": "query_preparer",
        "sources": "candidate_source",
        "fusion": "fuser",
        "passages": "passage_builder",
        "scorers": "passage_scorer",
        "assemblers": "context_assembler",
    }
    result: dict[str, list[dict[str, Any]]] = {}
    for kind_key in kind_order:
        kind = kinds_map[kind_key]
        items: list[dict[str, Any]] = []
        for pair, cls in COMPONENTS.items():
            if pair[0] != kind:
                continue
            info: dict[str, Any] = {
                "name": pair[1],
                "kind": kind,
                "requires": sorted(str(r) for r in _get_requires(cls)),
            }
            params = _get_params_schema(cls)
            if params is not None:
                info["params_schema"] = params.model_json_schema()
            else:
                info["params_schema"] = None
            items.append(info)
        result[kind_key] = sorted(items, key=lambda d: d["name"])
    return result
