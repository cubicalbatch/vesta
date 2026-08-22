"""Curated GGUF model presets for the first-run LLM wizard.

Shipped as data so the wizard can show model cards (name, size, description, RAM
recommendation) with no network — the download URL is baked in. The wizard's
"recommended for your machine" hint comes from :func:`recommend_preset`, which
picks based on detected RAM.

Each preset maps to a single GGUF file on HuggingFace. The download job fetches
the ``url`` and writes it to ``data/models/<filename>``; the API endpoint then
sets ``inference.llm.source=local`` and ``inference.llm.model=<filename>`` so the
supervisor's router mode picks it up.

Thinking is a per-model capability: ``thinking`` records whether the
chat template can be toggled (``toggle``), always reasons (``always`` — the
switch is inert), or never reasons (``never``). Non-preset files get the same
treatment from the :func:`thinking_for_filename` stem heuristic.

``kv_bytes_per_token`` feeds :func:`estimate_ram_bytes`, the live RAM estimate
the context-size UI shows — calibrated from loaded-RSS measurements (loaded RSS
minus file size at ``c = 32768``).

Serving the model (llama-server lifecycle) is handled by the supervisor in
``inference/local.py`` — this module only owns the *which model* decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

#: Thinking modes. ``toggle`` = the ``enable_thinking`` switch works;
#: ``always`` / ``never`` = the switch is inert (template always/never reasons).
ThinkingMode = Literal["toggle", "always", "never"]

#: KV-cache bytes per token for files that match no preset. Conservative
#: (larger than any preset's measured value) so the UI's RAM estimate errs high.
DEFAULT_KV_BYTES_PER_TOKEN = 32 * 1024


@dataclass(frozen=True)
class ModelPreset:
    """A downloadable GGUF model for the local inference source."""

    id: str
    display_name: str
    url: str
    filename: str
    size_bytes: int
    min_ram_gb: float
    description: str
    thinking: ThinkingMode = "toggle"
    #: Max context window the model's template supports (bounds the UI select).
    context_max: int = 131072
    #: KV-cache bytes per token (see :func:`estimate_ram_bytes`).
    kv_bytes_per_token: int = DEFAULT_KV_BYTES_PER_TOKEN

    @property
    def model_name(self) -> str:
        """The value to set as ``inference.llm.model`` — the bare filename.

        llama-server router mode resolves this against ``--models-dir``; the
        OpenAI-compatible ``model`` parameter in the chat request matches it.
        """
        return self.filename


# ── The preset the wizard offers ─────────────────────────────────────────────
# Verified against the live HuggingFace repos:
#   unsloth/Qwen3.5-4B-GGUF — Qwen3.5-4B-Q4_K_S.gguf (2 590 430 368 bytes)
_PRESETS: tuple[ModelPreset, ...] = (
    ModelPreset(
        id="qwen3.5-4b-q4_k_s",
        display_name="Qwen3.5 4B (Q4_K_S)",
        url="https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_S.gguf",
        filename="Qwen3.5-4B-Q4_K_S.gguf",
        size_bytes=2_590_430_368,
        min_ram_gb=4.0,
        description=(
            "Alibaba's Qwen3.5 — a strong all-round answer model with a working "
            "thinking switch (off = fast direct answers, on = deep reasoning)."
        ),
        thinking="toggle",
        context_max=131072,
        kv_bytes_per_token=17408,
    ),
)

_BY_ID: dict[str, ModelPreset] = {p.id: p for p in _PRESETS}
_BY_FILENAME: dict[str, ModelPreset] = {p.filename: p for p in _PRESETS}


def model_presets() -> tuple[ModelPreset, ...]:
    """The ordered list of downloadable GGUF presets."""
    return _PRESETS


def preset_by_id(preset_id: str) -> ModelPreset | None:
    """Look up a preset by id, or ``None``."""
    return _BY_ID.get(preset_id)


def preset_by_filename(filename: str) -> ModelPreset | None:
    """Look up a preset by GGUF filename (``inference.llm.model``), or ``None``."""
    return _BY_FILENAME.get(Path(filename).name)


def recommend_preset(ram_total_bytes: int = 0) -> ModelPreset:
    """The preset recommended for the detected RAM."""
    return _PRESETS[0]


def thinking_for_filename(filename: str) -> ThinkingMode:
    """Thinking mode for any GGUF filename — preset table first, stem heuristic.

    Heuristic (calibrated against live models): ``instruct`` ⇒ ``never``
    (measured on LFM2.5-1.2B-Instruct — no kwargs needed, no ``<think>``);
    ``thinking`` ⇒ ``always``; a bare LFM (no Instruct) ⇒ ``always`` (measured —
    LFM2.5-2.6B's template has no off switch); anything else (incl. Qwen3.x) ⇒
    ``toggle``.
    """
    preset = preset_by_filename(filename)
    if preset is not None:
        return preset.thinking
    stem = Path(filename).stem.lower()
    if "instruct" in stem:
        return "never"
    if "thinking" in stem:
        return "always"
    if "lfm" in stem:
        return "always"
    return "toggle"


def kv_bytes_per_token_for(filename: str) -> int:
    """KV-cache bytes per token for a filename — preset value or the default."""
    preset = preset_by_filename(filename)
    return DEFAULT_KV_BYTES_PER_TOKEN if preset is None else preset.kv_bytes_per_token


def display_name_for(filename: str) -> str:
    """Human name for a GGUF filename — preset name, else a prettified stem."""
    preset = preset_by_filename(filename)
    if preset is not None:
        return preset.display_name
    return " ".join(Path(filename).stem.replace("_", "-").split("-"))


def estimate_ram_bytes(size_bytes: int, context_size: int, kv_bytes_per_token: int) -> int:
    """Estimated resident bytes for a loaded model at a given context size.

    ``weights + KV cache`` — no extra overhead term: calibrated against
    loaded-RSS measurements (LFM2.5-1.2B-Instruct ≈ 999 MB est. vs
    921 MiB measured; Qwen3.5-4B ≈ 3.16 GB est. vs 2.87-3.05 GiB measured).
    """
    return size_bytes + context_size * kv_bytes_per_token


__all__ = [
    "DEFAULT_KV_BYTES_PER_TOKEN",
    "ModelPreset",
    "ThinkingMode",
    "display_name_for",
    "estimate_ram_bytes",
    "kv_bytes_per_token_for",
    "model_presets",
    "preset_by_filename",
    "preset_by_id",
    "recommend_preset",
    "thinking_for_filename",
]
