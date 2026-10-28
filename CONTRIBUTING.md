# Contributing

Thanks for taking an interest. Bug reports, fixes, new readers and embedders, better docs and real-world evaluation results all help. This page covers the setup, how the code is organised, what the tests expect and how the writing should read.

## Setup

```bash
git clone https://github.com/<you>/context-search-engine.git
cd context-search-engine
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dense,ocr,dev]"                    # on Linux install CPU torch first, see README
contextsearch doctor
pytest -q                                            # fast suite, no model download
pytest -q -m slow                                    # needs the default model on disk
ruff check . && ruff format --check .
```

Conda works the same way inside any environment. The fast test suite uses a hashing embedder (`test/hashing-64`) so it runs without torch; the slow tests use the real model and reranker.

To run the UI against your changes: `contextsearch serve --data-dir /tmp/cse-dev` and open http://127.0.0.1:8080. The UI is plain HTML, CSS and JavaScript in `contextsearch/ui/`; reload the page (Ctrl+Shift+R) after editing it.

## Where things are

| Path | What lives there |
|---|---|
| `contextsearch/config.py` | settings; the help text there becomes `docs/CONFIGURATION.md` |
| `contextsearch/store.py` | SQLite schema and every query |
| `contextsearch/readers/` | one file format per class |
| `contextsearch/chunking.py` | chunkers |
| `contextsearch/embedding.py`, `rerank.py` | model wrappers and presets |
| `contextsearch/retrieval/` | lexical and dense retrieval, fusion, the search pipeline |
| `contextsearch/ingest/` | scanning, indexing, the worker, the watcher |
| `contextsearch/engine.py` | the facade everything else calls |
| `contextsearch/api/` | FastAPI routes and the UI |
| `contextsearch/cli.py`, `evaluate.py` | the command line and the evaluation |
| `tests/` | one file per area; `conftest.py` has the fixtures and a tiny PDF writer |
| `docs/` | the reference pages linked from the README |

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) explains how the pieces fit; [docs/EXTENDING.md](docs/EXTENDING.md) shows how to add a reader, embedder, reranker or chunker.

## Making a change

1. Branch from `main`.
2. Keep the change focused. A reader and a UI tweak are two pull requests.
3. Run `ruff check . && ruff format .` and `pytest -q` before pushing. CI runs both on Linux and Windows and also builds the Docker image and runs the suite inside it.
4. If you touched `config.py`, regenerate the reference: `contextsearch config --markdown > docs/CONFIGURATION.md` (write it as UTF-8; CI checks that the file matches the code).
5. If you added or renamed an API route, update `docs/API.md`; a test checks every path in the OpenAPI schema is mentioned there.
6. Add a line to `CHANGELOG.md` under an `Unreleased` heading for anything a user would notice.

Commit messages: a short imperative summary line (`Add EML reader`, `Fix overlap when a sentence exceeds chunk_words`), details in the body if needed.

## Tests

Tests exist to catch regressions in behaviour that matters, not to raise a coverage number. Before adding one, ask what would break silently without it. Good candidates: an offset that must stay exact, a sync decision (new, changed, removed, retried), an error that must keep its shape, a fusion formula. Tests that only assert the code ran are not worth their maintenance.

- Fast tests must not download anything or need a GPU; use the hashing embedder and the `testing` profile (`tests/conftest.py`).
- Mark tests that need a real model with `@pytest.mark.slow`.
- Use temporary folders; never write into the repository.
- Prefer one clear assertion message over several vague asserts.

## Writing

Documentation, docstrings, comments, commit messages and UI text are written for people who will read them under time pressure. A few rules the project sticks to:

- Plain sentences. Say what a thing does, when to use it and what happens when it goes wrong. No marketing adjectives.
- Explain why in comments, not what. If the code needs a comment to explain what it does, change the code.
- No em dashes or en dashes; use commas, colons or full stops.
- No emoji in headings, tables or code.
- British or American spelling, either is fine, but be consistent within a file.
- Real numbers come from measurements (see `docs/PERFORMANCE.md`); do not invent figures for an example.

## Reporting a problem

Open an issue with: the output of `contextsearch doctor`, the version, the operating system and Python version, the log lines around the request id (for API problems), and what you expected instead. For search quality, include the query, what came back and what should have. For a crash while indexing, the file type and, if you can share it, the file.

## Code of conduct

Be kind and constructive. See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## License

By contributing you agree that your contribution is licensed under the MIT license in [LICENSE](LICENSE).
