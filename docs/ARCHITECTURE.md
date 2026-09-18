# Architecture

contextsearch is one Python process around one SQLite file. This page walks through the pieces, what the index looks like on disk, how folders are kept in sync, and how a search request moves through the system.

## Components

```
                 +-----------------------------+
   browser  <--> |  FastAPI app  (api/)         |  <--> curl, your code
                 |  routes, auth, error shape   |
                 +--------------+--------------+
                                |
                 +--------------v--------------+
                 |  Engine  (engine.py)         |   one per data folder
                 |  search, sources, uploads    |
                 +---+----------+-----------+--+
                     |          |           |
        +------------v--+  +---v-------+  +v-----------------+
        | Searcher      |  | Ingest    |  | Store (SQLite)   |
        | lexical/dense |  | worker    |  | docs, pages,     |
        | fusion/rerank |  | watcher   |  | chunks, FTS5,    |
        +-------+-------+  +-----+-----+  | vectors, events  |
                |                |        +------------------+
        +-------v-------+  +-----v-----+
        | VectorIndex   |  | Readers   |  pdf, docx, pptx, text, html, image (OCR)
        | per model     |  | Chunker   |
        +---------------+  +-----------+
                |
        +-------v-------+
        | Embedders     |  sentence-transformers, one instance per model
        | Rerankers     |  cross-encoders
        +---------------+
```

| Module | Job |
|---|---|
| `config.py` | Typed settings from `CSE_*` variables, profiles, the chunking signature, the generated configuration reference |
| `store.py` | Everything on disk: SQLite schema, documents, pages, chunks, FTS5, vectors per model, revisions, events, query log |
| `readers/` | One reader per format. A reader returns pages of plain text and a title; page numbers are real for PDF and PPTX |
| `chunking.py` | Cuts page text into chunks that are verbatim slices with character offsets |
| `embedding.py` | Embedding models behind one interface, presets for known model families, the query embedding cache |
| `vectors.py` | The in-memory matrix per model that answers cosine searches |
| `rerank.py` | Cross-encoder rerankers |
| `retrieval/` | The lexical and dense retrievers, fusion, and the pipeline that assembles results with evidence |
| `ingest/` | Folder scanning, the diff against the index, the ingestor, the background worker with jobs, the file watcher |
| `engine.py` | The facade that wires all of the above together; the library entry point |
| `api/` | FastAPI routes, request models, auth, error handlers, the bundled UI |
| `cli.py` | The `contextsearch` command |
| `evaluate.py` | Scoring strategies on labelled queries |
| `observability.py` | Logging, request ids, traces, metrics |

## The index on disk

`<data_dir>/contextsearch.db` is the whole index. Runs in WAL mode so searches never wait for a write.

| Table | Holds |
|---|---|
| `sources` | Watched folders and the uploads area: path, include and exclude rules, last scan summary |
| `documents` | One row per file: size, mtime, content hash, revision, status (`indexed`, `failed`, `removed`, `pending`), page and chunk counts, title, the chunking signature the chunks were cut with |
| `revisions` | One row per indexed revision of a document: hash, size, pages, chunks, when |
| `pages` | The normalised text of every page. This is what the viewer shows and what chunk offsets point into |
| `chunks` | Verbatim slices of pages: page, character range, word count, text. Ids are never reused |
| `chunks_fts` | FTS5 index over chunk text with the porter stemmer, kept in sync by triggers. Provides BM25 ranking and highlighting |
| `models` | Embedding models with a dense index: name and dimension |
| `embeddings` | One float32 vector per (chunk, model), so several models can be indexed side by side |
| `events` | What happened to which file: added, updated, removed, failed, scan summaries, model builds |
| `queries` | Recent searches with strategy, latency and result count |

Removing a file keeps its `documents` row with status `removed` and drops its pages, chunks and vectors, so the history stays readable without keeping the content. Removing a source deletes everything it indexed.

Vectors are stored as blobs in SQLite and loaded into a numpy matrix per model at startup (a few seconds per million vectors). Search is an exact dot product against that matrix, which is deterministic and fast enough for corpora up to a few million chunks. See [PERFORMANCE.md](PERFORMANCE.md) for the numbers.

## From file to chunk

1. The reader for the extension produces pages. PDF text comes from PDFium (`pypdfium2`); a page with no text layer goes to OCR when `CSE_OCR` is on. DOCX paragraphs and tables are read in document order. PPTX gives one page per slide, including the notes. HTML is stripped of scripts and styles.
2. Every page is normalised: line endings, runs of spaces, NFC unicode. Paragraph breaks are kept.
3. The chunker cuts each page. The default `recursive` strategy packs sentence-like units up to `chunk_words` (200), prefers to end on a sentence, and carries about `chunk_overlap_words` (30) into the next chunk. A chunk is always `page_text[char_start:char_end]`, so a result can be traced to an exact span and highlighted.
4. Chunks are written to SQLite in one transaction together with the pages and the new document revision. FTS5 is updated by triggers in the same transaction.
5. Each embedding model with a dense index encodes the chunk texts in batches and the vectors are stored and added to the live matrix. If a model fails, keyword search still has everything; the missing vectors show up in `contextsearch models` and can be filled in later.

Because chunk offsets point into the stored page text, the viewer can open the page and mark the exact passage a result came from:

<p align="center"><img src="images/viewer.jpg" alt="The page viewer with the matched passage and query terms marked" width="900" /></p>

The chunking settings form a signature stored with every document. When the settings change, `contextsearch reindex` (or the Reindex button) re-cuts only the documents chunked with older settings.

## Keeping folders in sync

The scan is the source of truth; file system events only make it faster.

A scan walks the folder with `os.scandir`, reading directory entries and stats but no file contents, so a rescan of a large tree with nothing new takes seconds. Every file seen is compared with the stored row:

| On disk | Stored | Action |
|---|---|---|
| present | not stored | index as a new document |
| same size and mtime | indexed with the current chunking signature | nothing |
| different size or mtime | indexed | hash the content; if the hash is unchanged only the stat is updated, otherwise re-index as a new revision |
| present | failed last time | try again (a locked file on Windows, a transient read error) |
| present | chunked with an older signature | re-index |
| missing | indexed | mark removed, drop its chunks and vectors |

Two guard rails: a folder that cannot be listed (permissions, unmounted share) never turns into a mass deletion, and files larger than `max_file_mb` are reported rather than read.

With `watch_folders` on (the default), `watchdog` subscribes to file system events for every source. Events are debounced per path for `watch_debounce_seconds`, since editors and copy tools write in bursts, and then handed to the worker: a created or modified file is indexed, a deleted or moved-away file is removed, a folder-level change triggers a scan of the source. Every `scan_interval_seconds` (5 minutes by default) each enabled source is rescanned anyway, which is what covers network shares and the cases where events do not arrive.

## The worker and jobs

All writes go through one background thread with a queue. That keeps SQLite happy (one writer), keeps memory bounded (one embedding batch in flight) and makes progress easy to report. The queue coalesces repeated events for the same file until it starts.

Jobs are `sync` (scan a source), `file` (index or remove one file), `build_model` (embed every chunk that lacks a vector for a model) and `reindex`. Each job carries progress (done, total, current file, counts by outcome), an error list and a result; they are visible at `/api/v1/jobs`, in the Activity page and in `contextsearch jobs`. A running job can be cancelled; it stops after the current file.

Uploads through the API land in `<data_dir>/uploads/<date>/` and go through the same path as a watched file. By default the upload call waits for indexing to finish so the file is searchable when the response comes back.

## A search request

```
GET /api/v1/search?q=...          request id assigned, appears in the log and the response
  Searcher.search
    lexical    FTS5 MATCH, AND first, OR to fill up           (2 to 20 ms)
    dense:M    embed query (cached), matrix dot product        (per model)
    fusion     RRF or weighted                                 (< 1 ms)
    rerank     cross-encoder over the top candidates           (optional, 100 to 500 ms on CPU)
    assemble   load chunks and documents, snippets, confidence, grouping
  query log row, metrics, log line with timings
```

Each stage is a span in the trace returned with the response, so a slow request explains itself. [SEARCH.md](SEARCH.md) describes what happens inside each stage and how the numbers are produced.

## Observability

- Logs go to stderr as text (development) or JSON (production), optionally to a file. Every line carries the request id of the request or job it belongs to.
- `/api/v1/system/metrics` exposes Prometheus text: request counts and latency histograms per method, searches per strategy, documents indexed by outcome, ingest time, documents and chunks gauges.
- `/api/v1/health` reports the components: database, dense availability and device, models and vector counts, watcher, worker and its current job.
- `/api/v1/system/events` and the query log show what the index did and what people searched for.

## Security model

contextsearch is built for an internal network. `CSE_API_KEY` protects every API route except the short form of `/health`. Folder sources must sit inside `CSE_ALLOWED_ROOTS` in production. Files are served only by document id, never by a client-supplied path. Uploads are size and type limited and stored under safe file names. The UI keeps the API key in the browser's local storage and sends it as a header; file links append it as a query parameter so they open in a new tab. Responses carry `X-Content-Type-Options`, `X-Frame-Options` and `Referrer-Policy`. See [SECURITY.md](../SECURITY.md).
