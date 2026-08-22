# Vesta production image.
#
# Multi-stage: the Svelte SPA, the Python runtime deps, the bundled llama.cpp
# (CPU variants + Vulkan), and the three default encoder models are each built
# in a throwaway stage; none of the build toolchain reaches the final image.
# The result is a single self-contained container — built frontend, the Python
# app, all runtime deps, a working local-LLM runtime, and the default encoders —
# that boots to the welcome page and answers over a local model once the user
# adds a GGUF, with no other downloads.
#
#   docker compose up -d --build   # serves on http://127.0.0.1:5129

# ── Stage 1: build the SPA ─────────────────────────────────────────────────
# adapter-static writes straight to ../src/vesta/static/app (vite.config.ts), so
# we lay the frontend out at /build/frontend and let it write to /build/src/….
FROM node:22-slim AS frontend
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./frontend/
RUN --mount=type=cache,target=/root/.npm \
    cd frontend && npm ci
COPY frontend/ ./frontend/
RUN mkdir -p src/vesta/static/app \
    && cd frontend && npm run build

# ── Stage 2: install Python runtime deps with uv ───────────────────────────
FROM python:3.13-slim AS deps
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1
RUN pip install --no-cache-dir uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# ── Stage 3: fetch llama.cpp (CPU variants + Vulkan backend) ───────────────
# The official `ubuntu-vulkan-x64` prebuilt is a superset: every CPU backend
# variant (sse42 … zen4, dispatched at runtime by feature scoring → no SIGILL on
# any x86-64 host) AND libggml-vulkan.so. Pinned tag — bumping is a release
# change. Building from source would add ~15 min and a toolchain for no gain.
FROM debian:bookworm-slim AS llamacpp
ARG LLAMA_TAG=b10373
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl tar \
    && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL -o /tmp/llc.tgz \
        "https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_TAG}/llama-${LLAMA_TAG}-bin-ubuntu-vulkan-x64.tar.gz" \
    && mkdir -p /opt/llama.cpp \
    && tar xzf /tmp/llc.tgz -C /opt/llama.cpp --strip-components=1 \
    && rm /tmp/llc.tgz \
    && chmod +x /opt/llama.cpp/llama-server

# ── Stage 4: bake the three default encoder models ─────────────────────────
# The default `standard`/`hybrid` profiles need the static + rerank (and embed
# for semantic indexing) models present. These are public ONNX files on HF, so
# no torch/optimum — just the files the encoder manager actually reads
# (onnx graph + external data + tokenizer.json). Dropped into a read-only image
# path; the entrypoint symlinks them into the data volume on first run.
FROM python:3.13-slim AS models
RUN pip install --no-cache-dir huggingface_hub
WORKDIR /opt/vesta/models
COPY <<'PY' /tmp/bake_models.py
from huggingface_hub import snapshot_download

jobs = {
    "minishlab/potion-retrieval-32M": ["onnx/model.onnx", "tokenizer.json"],
    "onnx-community/granite-embedding-small-english-r2-ONNX": [
        "onnx/model_quantized.onnx",
        "onnx/model_quantized.onnx_data",
        "tokenizer.json",
    ],
    "Xenova/ms-marco-MiniLM-L-6-v2": ["onnx/model_quantized.onnx", "tokenizer.json"],
}
for repo, pats in jobs.items():
    snapshot_download(repo, local_dir=repo, allow_patterns=pats)
    print(f"baked {repo}")
PY
RUN python /tmp/bake_models.py

# ── Stage 5: runtime ───────────────────────────────────────────────────────
FROM python:3.13-slim AS runtime
# llama.cpp runtime: the bundled libs live in /opt/llama.cpp (llama-server links
# libllama/libggml/libmtmd there). The cpu/vulkan backends are found by ggml's
# default search — the executable's directory — no GGML_BACKEND_PATH.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    PATH=/app/.venv/bin:$PATH \
    LD_LIBRARY_PATH=/opt/llama.cpp
# llama-server is resolved by the app's default `inference.local.binary_path`
# value ("llama-server") via PATH — a symlink, not a dotted ENV var: the sh
# entrypoint (dash) drops env vars whose names aren't valid shell identifiers,
# so a dotted ENV never reaches uvicorn.
# libgomp1 → onnxruntime; libvulkan1 + mesa-vulkan-drivers → the Vulkan loader
# and the Intel/AMD ICDs so a GPU works out of the box when /dev/dri is exposed
# (see docker-compose.yml). No GPU/device → ggml-vulkan finds zero devices and
# silently falls back to CPU; never crashes.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 libvulkan1 mesa-vulkan-drivers \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=llamacpp /opt/llama.cpp /opt/llama.cpp
RUN ln -s /opt/llama.cpp/llama-server /usr/local/bin/llama-server
COPY --from=deps /app/.venv /app/.venv
COPY --from=models /opt/vesta/models /opt/vesta/models
COPY pyproject.toml uv.lock ./
COPY src ./src
COPY --from=frontend /build/src/vesta/static/app ./src/vesta/static/app
COPY docker-entrypoint.sh /app/docker-entrypoint.sh

EXPOSE 8080
ENTRYPOINT ["/app/docker-entrypoint.sh"]
# §13.3: one uvicorn worker, ever. Host 0.0.0.0 so compose can map the port.
CMD ["uvicorn", "vesta.main:app", "--host", "0.0.0.0", "--port", "8080"]
