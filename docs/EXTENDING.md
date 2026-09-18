# Using contextsearch as a library and extending it

Everything the server does is available from Python through the `Engine` class, and each stage (reading files, chunking, embedding, reranking) is a small interface you can replace.

## Library use

```python
from contextsearch import Engine, Settings

settings = Settings.from_env(data_dir="./data", embedding_models="BAAI/bge-small-en-v1.5", log_level="WARNING")
engine = Engine(settings, configure_logs=False)   # your app owns logging
engine.start()                                    # loads indexes, starts the worker and the watcher

source, job = engine.add_folder("/srv/docs", name="Shared drive")
job.wait()                                        # or poll job.progress / job.status
print(job.result)                                 # {"added": 120, "updated": 0, "removed": 0, ...}

response = engine.search("refund for a late delivery", strategy="hybrid", top_k=5, rerank=True)
for hit in response.hits:
    print(hit.rank, hit.confidence_label, hit.document.relpath, "page", hit.chunk.page)
    print("   ", hit.snippet)
    print("   ", hit.evidence["found_by"], hit.evidence["lexical"], hit.evidence["dense"])
print(response.trace.timings())                  # {"lexical": 2.1, "dense:BAAI/bge-small-en-v1.5": 14.9, ..., "total": 20.8}

engine.close()
```

`Engine` is also a context manager (`with Engine.open("./data") as engine:`). `Engine.open(data_dir, **overrides)` builds settings from the environment plus the overrides you pass.

Useful methods:

| Method | What it does |
|---|---|
| `search(query, **options)` | Options are the fields of `SearchRequest`: `strategy`, `models`, `top_k`, `candidates`, `fusion`, `alpha`, `rerank`, `rerank_model`, `group_by_document`, `source_id`, `ext`, `document_id`, `min_confidence`, `prefix_last`, `snippet_chars`, `explain`. Returns a `SearchResponse` with `hits`, `retrievers`, `trace`, `warnings`; `to_dict()` gives the API JSON |
| `compare(query, variants)` | One query under up to four option sets |
| `add_folder(path, name, recursive, include, exclude)` | Registers a source, starts watching it, queues a scan. Returns `(source, job)` |
| `sync_source(id)`, `sync_all()` | Queue scans |
| `remove_source(id)`, `set_source_enabled(id, bool)` | |
| `save_upload(filename, stream, wait)` | Store a file in the uploads area and index it |
| `delete_document(id)` | Uploaded documents only |
| `build_dense_index(model)` | Queue embedding of every chunk with another model; returns the job |
| `drop_dense_index(model)` | |
| `reindex(force)` | Re-chunk after changing chunking settings |
| `document_detail(id)`, `document_page(id, page)`, `document_file(id)` | What the API returns for a document |
| `health()`, `stats()`, `models_info()` | |
| `store` | The `Store` for direct queries (`list_documents`, `list_events`, `recent_queries`, ...) |

Everything that indexes runs on the engine's worker thread. `start(workers=False)` opens the index read-only-ish for one-shot searches (this is what `contextsearch search` does) without starting any background thread.

Settings come from `Settings.from_env()`; pass `env={}` and `dotenv=False` to ignore the process environment, as the tests do. `Settings.markdown_reference()` renders the configuration page.

### Evaluating from code

```python
from contextsearch.evaluate import StrategySpec, evaluate, format_table, load_labels

labels = load_labels("labels.json")
specs = [StrategySpec.parse(s) for s in ("lexical", "hybrid", "hybrid+rerank", "dense:intfloat/e5-base-v2")]
reports = evaluate(engine, labels, specs, k=10)
print(format_table(reports))
print([r.to_dict() for r in reports])
```

## Adding a reader

A reader turns one file into pages of text. Register one for a new extension, or replace a built-in one (for example to use a different PDF library):

```python
from pathlib import Path
from contextsearch.readers import Page, ParsedDocument, Reader, register

class EmlReader(Reader):
    name = "eml"
    extensions = ("eml",)

    def read(self, path: Path, settings) -> ParsedDocument:
        import email
        message = email.message_from_bytes(path.read_bytes())
        body = message.get_payload(decode=True) or b""
        text = f"Subject: {message['subject']}\nFrom: {message['from']}\n\n{body.decode('utf-8', 'replace')}"
        return ParsedDocument(pages=[Page(number=1, text=text)], title=message["subject"] or "")

register(EmlReader())
```

Do this before `engine.start()`, and add `eml` to `CSE_INCLUDE_EXTENSIONS` so scans pick the files up. Pages are numbered from 1; formats without pages return one page. Raise `contextsearch.errors.ReadError` for a file you cannot parse; the scan records it as failed and retries next time. Text is normalised after your reader returns, so do not worry about line endings or unicode forms. Keep the text in reading order: chunk offsets point into it and the viewer shows it as is.

## Adding an embedding model

Any sentence-transformers model needs no code: `contextsearch build org/model` or `CSE_EMBEDDING_MODELS=org/model`. If the model card asks for prompts, add them with `CSE_EMBEDDING_QUERY_PREFIX` and `CSE_EMBEDDING_DOCUMENT_PREFIX`, or add a preset to `PRESETS` in `contextsearch/embedding.py` so the prefixes travel with the model name.

For a model that sentence-transformers cannot load (an API, ONNX through another runtime, an image model), implement `Embedder`:

```python
import numpy as np
from contextsearch.embedding import Embedder

class MyEmbedder(Embedder):
    name = "myorg/my-model"

    @property
    def dimension(self) -> int:
        return 512

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        vectors = ...                              # shape (len(texts), 512)
        return (vectors / np.linalg.norm(vectors, axis=1, keepdims=True)).astype(np.float32)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self.embed_documents(texts)

engine = Engine(settings)
engine.embedders.register(MyEmbedder())
engine.start()
engine.build_dense_index("myorg/my-model").wait()
```

Return unit-length float32 vectors: cosine similarity is a plain dot product everywhere else. The index for a model is keyed by `name`, so the same name must be registered before `start()` on every run.

Image embeddings fit the same shape. A CLIP-style embedder that encodes an image reader's output (or the image file itself) and encodes text queries into the same space is the planned route for searching pictures; the store already keeps one vector per (chunk, model) and the pipeline does not care what produced them.

## Adding a reranker

```python
import numpy as np
from contextsearch.rerank import Reranker

class MyReranker(Reranker):
    name = "myorg/my-reranker"

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        return np.asarray([...], dtype=np.float32)   # one value in 0..1 per passage

engine.rerankers.register(MyReranker())
engine.search("...", rerank=True, rerank_model="myorg/my-reranker")
```

## Adding a chunker

`contextsearch.chunking.CHUNKERS` maps a name to a `Chunker` whose `split(text, chunk_words, overlap_words, min_words)` returns `(char_start, char_end)` spans covering the text. The contract is that `text[start:end]` is the chunk, verbatim, with no leading or trailing whitespace. Register yours under a new name, set `CSE_CHUNKER` to it and run `contextsearch reindex`; the name is part of the chunking signature so existing documents are re-cut.

## Adding an API route

`contextsearch.api.create_app()` returns a normal FastAPI app. Mount your own router on it, reuse `get_engine` and `require_api_key` from `contextsearch.api.deps`, and raise the exceptions in `contextsearch.errors` to get the standard JSON error shape:

```python
from fastapi import APIRouter, Depends
from contextsearch.api import create_app
from contextsearch.api.deps import get_engine, require_api_key

router = APIRouter(prefix="/api/v1/mine", dependencies=[Depends(require_api_key)])

@router.get("/popular-queries")
def popular_queries(engine=Depends(get_engine)):
    counts: dict[str, int] = {}
    for row in engine.store.recent_queries(1000):
        counts[row["query"]] = counts.get(row["query"], 0) + 1
    return {"queries": sorted(counts.items(), key=lambda kv: -kv[1])[:20]}

app = create_app()
app.include_router(router)
```

Run it with `uvicorn myapp:app --port 8080` (one worker).

## Running only parts

The pieces work on their own for scripts and notebooks:

```python
from contextsearch.readers import read_document, register_builtin_readers
from contextsearch.chunking import chunk_document

register_builtin_readers(settings)
doc = read_document("report.pdf", settings)
for chunk in chunk_document(doc, settings):
    print(chunk.page, chunk.char_start, chunk.char_end, chunk.text[:60])
```
