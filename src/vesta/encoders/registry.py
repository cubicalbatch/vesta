"""Model registry — every known model's runtime shape, prefixes, pooling and
normalization, owned in exactly one place.

By keeping ``query_prefix``/``passage_prefix``/``pooling``/``normalize`` here
instead of in caller code, there is nothing for a caller to forget (e.g. an
asymmetric-prefix model missing its query prefix). Vesta's shipped defaults
(Granite, model2vec/potion) need no prefixes at all; the field still exists so a
prefix model (e.g. Nomic, BGE) can be added safely later.

Three runtime *kinds*, because the three model roles are distinct:

* ``bi_encoder_padded`` — a padded-batch transformer forward pass
  (``input_ids``/``attention_mask`` → pooled sentence embedding). Used by
  ``embed`` (also reused for bulk indexing).
* ``bi_encoder_ragged`` — a model2vec-style token->vector table export: an
  ``EmbeddingBag``-shaped ONNX graph (``input_ids`` flat + ``offsets`` → pooled
  embedding), no attention, effectively zero FLOPs. Used by ``static``.
* ``cross_encoder_pair`` — a sequence-pair classification forward pass
  (query, passage) -> one logit. Used by ``rerank``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

#: A model role — distinct from "kind" (the runtime shape). Three roles:
#: ``static`` is Stage B1's shortlister, ``embed`` is the
#: indexing bi-encoder (also usable query-side), ``rerank`` is Stage B2's
#: cross-encoder.
Role = Literal["static", "embed", "rerank"]
RuntimeKind = Literal["bi_encoder_padded", "bi_encoder_ragged", "cross_encoder_pair"]


@dataclass(frozen=True)
class ModelSpec:
    """Everything needed to load and correctly use one model.

    ``onnx_data_file`` is set for models whose ``.onnx`` external-data sidecar
    must be downloaded alongside the graph (>2 GB protobuf limit workaround;
    Granite's quantized export splits this way). ``max_tokens`` is the model's
    architectural ceiling for ``embed``/``static``; for ``rerank`` it is Vesta's
    chosen operating point (256), not the model's true ceiling
    (512) — the truncation is a deliberate latency policy, not a limitation.
    """

    repo_id: str
    role: Role
    kind: RuntimeKind
    onnx_file: str
    onnx_data_file: str | None
    dim: int  # 0 for cross-encoders — no embedding space
    max_tokens: int
    query_prefix: str
    passage_prefix: str
    normalize: bool
    pooling: Literal["cls", "mean"]
    license: str


#: Known models, keyed by HF repo id. Settings (``encoders.<role>.model``) hold
#: a repo id from this table; an unknown id degrades the role's capability off
#: (nothing to load) rather than raising.
MODEL_SPECS: dict[str, ModelSpec] = {
    "minishlab/potion-retrieval-32M": ModelSpec(
        repo_id="minishlab/potion-retrieval-32M",
        role="static",
        kind="bi_encoder_ragged",
        onnx_file="onnx/model.onnx",
        onnx_data_file=None,
        dim=512,
        max_tokens=8192,  # unbounded in practice (table lookup + mean)
        query_prefix="",
        passage_prefix="",
        normalize=True,
        pooling="mean",
        license="mit",
    ),
    "onnx-community/granite-embedding-small-english-r2-ONNX": ModelSpec(
        repo_id="onnx-community/granite-embedding-small-english-r2-ONNX",
        role="embed",
        kind="bi_encoder_padded",
        onnx_file="onnx/model_quantized.onnx",
        onnx_data_file="onnx/model_quantized.onnx_data",
        dim=384,
        max_tokens=512,  # chunking operating ceiling
        query_prefix="",
        passage_prefix="",
        normalize=True,
        pooling="mean",
        license="apache-2.0",
    ),
    "onnx-community/gte-modernbert-base-ONNX": ModelSpec(
        repo_id="onnx-community/gte-modernbert-base-ONNX",
        role="embed",
        kind="bi_encoder_padded",
        onnx_file="onnx/model_quantized.onnx",
        onnx_data_file="onnx/model_quantized.onnx_data",
        dim=768,
        max_tokens=512,
        query_prefix="",
        passage_prefix="",
        normalize=True,
        pooling="mean",
        license="apache-2.0",
    ),
    "Xenova/ms-marco-MiniLM-L-6-v2": ModelSpec(
        repo_id="Xenova/ms-marco-MiniLM-L-6-v2",
        role="rerank",
        kind="cross_encoder_pair",
        onnx_file="onnx/model_quantized.onnx",
        onnx_data_file=None,
        dim=0,
        max_tokens=256,  # deliberate truncation, not the model's ceiling
        query_prefix="",
        passage_prefix="",
        normalize=False,
        pooling="cls",  # BERT [CLS] pooled classifier head; informational only
        license="apache-2.0",
    ),
}

#: The recommended default repo per role. Settings default to these.
DEFAULT_MODEL: dict[Role, str] = {
    "static": "minishlab/potion-retrieval-32M",
    "embed": "onnx-community/granite-embedding-small-english-r2-ONNX",
    "rerank": "Xenova/ms-marco-MiniLM-L-6-v2",
}


def resolve_spec(repo_id: str, role: Role) -> ModelSpec | None:
    """Look up a model by repo id, validating it matches the expected role.

    Returns ``None`` for an unknown repo id or a role mismatch (e.g. an
    ``embed`` model configured for ``encoders.rerank.model``) — the manager
    treats this exactly like a missing file: the capability stays off rather
    than the process raising on a typo'd setting.
    """
    spec = MODEL_SPECS.get(repo_id)
    if spec is None or spec.role != role:
        return None
    return spec


__all__ = ["DEFAULT_MODEL", "MODEL_SPECS", "ModelSpec", "Role", "RuntimeKind", "resolve_spec"]
