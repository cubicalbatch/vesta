"""Model fetch helper — dev/install-time only, never on the request path.

Models can be baked into the image at ``/opt/vesta/models/`` or downloaded locally.
This utility populates ``encoders.model_dir``: a thin wrapper over
``huggingface_hub.hf_hub_download`` that lays files out exactly the way
:class:`~vesta.encoders.manager.EncoderManager` expects to find them
(``model_dir/<repo_id>/<relative path>``, mirroring the repo's own tree).

Deliberately **not** called from ``EncoderManager`` or any capability probe —
the runtime path only ever reads what is already on disk (no network I/O in
the request path of an offline-first appliance) and degrades the role's
capability off when it is not there yet.
"""

from __future__ import annotations

from pathlib import Path

from vesta.encoders.registry import ModelSpec


def ensure_model(spec: ModelSpec, model_dir: Path) -> Path:
    """Download every file a role needs into ``model_dir/<repo_id>/...``.

    Idempotent — ``hf_hub_download`` no-ops (cache hit) if the file is already
    present with a matching hash. Returns the local directory the model was
    written to. Requires network access; never call this from a request handler.
    """
    from huggingface_hub import hf_hub_download

    dest = model_dir / spec.repo_id
    dest.mkdir(parents=True, exist_ok=True)
    filenames = [spec.onnx_file, "tokenizer.json"]
    if spec.onnx_data_file:
        filenames.append(spec.onnx_data_file)
    for filename in filenames:
        hf_hub_download(spec.repo_id, filename, local_dir=dest)
    return dest


__all__ = ["ensure_model"]
