# Vesta

**A private, offline library you can ask questions.**

Vesta turns Kiwix ZIM archives — Wikipedia, StackExchange, Project Gutenberg,
Wikivoyage, and thousands more — into a knowledge base you search and converse
with. Ask a question in plain language and get a direct answer, streamed as
it's written, with every claim cited back to the passage it came from. It all
runs on your machine: after setup, no internet connection is needed.

## Highlights

- **Ask, don't dig.** Get a direct answer instead of a list of links.
- **Answers you can verify.** Every claim carries a citation to the exact
  source passage. Click through and check it yourself.
- **Truly offline.** Once set up, Vesta
  works air-gapped and nothing ever leaves your machine.
- **Your hardware.** Run the language model locally on CPU or GPU,
  or point Vesta at any OpenAI-compatible endpoint you already have.
- **One container.** App, models, and model runtime ship as a
  single Docker image with a guided first-run setup.

## Quick start

The fastest way in is Docker:

```bash
docker compose up -d --build
```

Then open **http://localhost:5129**. On first run, a setup wizard walks you
through downloading an archive and a model, from there everything happens in
the UI.

All persistent data (archives, models, database) lives in a single directory
mounted from the host, so backing up or moving Vesta is a file copy.

### Without Docker

You'll need Linux or macOS, Python 3.13+ with [uv](https://docs.astral.sh/uv/),
and Node 20+ to build the UI.

```bash
git clone https://github.com/cubicalbatch/vesta.git
cd vesta
uv sync                                  # Python dependencies
uv run vesta models                      # search models
cd frontend && npm ci && npm run build && cd ..   # build the UI
./start.sh                               # serve on http://localhost:5586
```

## Building your library

Two ways to add archives:

- **In the app**: browse the built-in Kiwix catalog and download whatever you
  want.
- **By hand**: drop `.zim` files into `data/zims/`; Vesta picks them up
  automatically.

Semantic search needs a one-time index build per archive — from the web UI,
or:

```bash
uv run vesta index              # all archives
uv run vesta index --depth 3    # highest answer quality
```

Index depth is a speed/quality dial: 1 builds fastest, 3 gives the best
answers but takes a lot to build.

## Choosing a model

Switchable at any time in Settings:

- **Local (default)** — drop a GGUF file into `data/models/` or download one
  in the UI. Vesta manages the model server for you: it loads on demand,
  unloads when idle, and uses your GPU when one is available.
- **Remote** — point Vesta at any OpenAI-compatible endpoint (Ollama, vLLM,
  OpenRouter, …).

## Configuration

Nearly everything is a setting in the web UI, and nearly everything applies
without restarts. 

## License

Apache 2.0
