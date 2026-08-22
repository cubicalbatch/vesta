#!/bin/sh
# Vesta container entrypoint.
#
# One job: make the image-baked encoder models appear inside the bind-mounted
# data volume so the app (which reads data/models/<repo_id>) sees them on first
# run, with zero download — while keeping the volume writable for models the
# user adds later. We do that with symlinks: data/models/<org>/<repo> → the
# read-only baked copy in the image. If a real file/dir already exists at a
# target (user downloaded their own), it is left alone. Then we exec the app.
#
# llama.cpp + Vulkan wiring is done via Dockerfile ENV (LD_LIBRARY_PATH,
# inference.local.binary_path); nothing to do here.
set -e

BAKED_MODELS="/opt/vesta/models"
VOLUME_MODELS="${VESTA_MODELS_DIR:-/app/data/models}"

if [ -d "$BAKED_MODELS" ]; then
    mkdir -p "$VOLUME_MODELS"
    for org_dir in "$BAKED_MODELS"/*; do
        [ -d "$org_dir" ] || continue
        org="$(basename "$org_dir")"
        mkdir -p "$VOLUME_MODELS/$org"
        for repo_dir in "$org_dir"/*; do
            [ -d "$repo_dir" ] || continue
            repo="$(basename "$repo_dir")"
            target="$VOLUME_MODELS/$org/$repo"
            if [ ! -e "$target" ]; then
                ln -s "$repo_dir" "$target"
            fi
        done
    done
fi

exec "$@"
