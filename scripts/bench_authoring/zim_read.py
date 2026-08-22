#!/usr/bin/env python3
"""Read (and sample) articles from the pinned Wikipedia ZIM for benchmark authoring.

Agents authoring benchmark questions use this to VERIFY that a fact actually
exists in the archive before recording it. Usage:

    uv run python scripts/bench_authoring/zim_read.py --title "Albert Einstein"
    uv run python scripts/bench_authoring/zim_read.py --path  "Albert_Einstein"
    uv run python scripts/bench_authoring/zim_read.py --sample 40

The article text printed is the SAME extraction the benchmark's retrieval and
answer pipeline sees (resiliparse main-content, infoboxes kept), so a fact
present in this text is a fact RAG can actually retrieve.
"""

from __future__ import annotations

import argparse
import sys

from libzim.reader import Archive

from vesta.zim.extract import extract_article
from vesta.zim.reader import read_entry_sync

ZIM = "data/zims/wikipedia_en_top_nopic_2026-06.zim"


def _extract_text(archive: Archive, path: str) -> tuple[str, str, str]:
    """Return (resolved_path, title, text) for one article path."""
    raw = read_entry_sync(archive, path)
    if raw.is_redirect:
        target = raw.redirect_target or "(redirect)"
        return target, raw.title, ""
    art = extract_article(raw.content, path=path, title=raw.title)
    return raw.path, raw.title, art.text


def cmd_read(args: argparse.Namespace) -> int:
    a = Archive(ZIM)
    path = args.path
    if not path and args.title:
        try:
            path = a.get_entry_by_title(args.title).path
        except Exception as exc:  # title not found
            print(f"NOT_FOUND title={args.title!r}: {exc}", file=sys.stderr)
            return 1
    if not a.has_entry_by_path(path):
        print(f"NOT_FOUND path={path!r}", file=sys.stderr)
        return 1
    p, t, text = _extract_text(a, path)
    print(f"# path: {p}")
    print(f"# title: {t}")
    print(text)
    return 0


def cmd_sample(args: argparse.Namespace) -> int:
    a = Archive(ZIM)
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    attempts = 0
    while len(out) < args.count and attempts < args.count * 40:
        attempts += 1
        e = a.get_random_entry()
        item = e.get_item()
        if item.mimetype != "text/html":
            continue
        if e.is_redirect:
            continue
        path = e.path
        if path in seen:
            continue
        seen.add(path)
        out.append((path, e.title or path))
    for path, title in sorted(out):
        print(f"{path}\t{title}")
    print(f"# sampled {len(out)} articles (requested {args.count})", file=sys.stderr)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("read", help="Print one article's extracted text.")
    r.add_argument("--title", help="Look up by human title, e.g. 'Albert Einstein'.")
    r.add_argument("--path", help="Look up by ZIM path, e.g. 'Albert_Einstein'.")
    r.set_defaults(func=cmd_read)
    s = sub.add_parser("sample", help="Sample N random article paths + titles.")
    s.add_argument("--count", type=int, default=40)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_sample)
    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
