# Changelog

All notable changes to this project are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and the project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.0] - 2026-09-18

Version 3 is a rewrite of the version 2 proof of concept into a package that can be run as a service, used as a library and extended. The data written by version 2 (`faiss_index.idx`, `index_to_chunk.pkl`, `document_metadata.pkl`, `app_config.json`) cannot be read by version 3; re-add the folder or re-upload the files. [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) has the migration notes.

### Added

- Hybrid search: BM25 over SQLite FTS5 plus embedding search, fused with reciprocal rank fusion (or a weighted blend), with an optional cross-encoder reranking stage.
- Evidence on every result: which retrievers found it and at what rank, matched and covered query terms, cosine per model, reranker score, a confidence value with a label, the page and character span in the source, and per-stage timings with a request id.
- Folder sources: scan a tree, watch it for changes (watchdog) with a periodic rescan as the safety net, new revisions for edited files, removal for deleted files, retries for locked or broken files, an allow-list of roots.
- Readers for PDF (PDFium), DOCX, PPTX (slides and notes), plain text and Markdown, HTML, CSV, JSON and other text formats. OCR for images and scanned PDF pages through RapidOCR (`ocr` extra).
- Sentence-aware chunking with exact character offsets, a chunking signature per document and `contextsearch reindex` to re-cut after settings change.
- Several embedding models side by side: any sentence-transformers model, dimension read from the model, presets with the right query prefixes for the bge, e5, nomic and arctic families, a Compare page and `contextsearch eval` (recall, MRR, nDCG, latency on a labels file).
- A REST API under `/api/v1` with an OpenAPI page at `/docs`, JSON errors with stable codes and request ids, API key auth, CORS settings, background jobs with progress and cancellation, uploads, document and page endpoints, Prometheus metrics, health and stats.
- A new web UI: search with search-as-you-type, ranking controls, the trace of every search, a page viewer that highlights the exact passage, Compare, Sources, Documents with revision history, Activity, light and dark themes, no build step and no external assets.
- A CLI: `serve`, `index`, `sync`, `search`, `sources`, `documents`, `models`, `build`, `drop`, `reindex`, `eval`, `doctor`, `config`, `db`.
- Typed configuration from `CSE_*` environment variables with development, production and testing profiles; `docs/CONFIGURATION.md` is generated from it.
- A Docker image and compose file, a CI workflow on Linux and Windows, a test suite, and documentation for architecture, search, API, deployment, performance, extending and troubleshooting.

### Changed

- The default embedding model is `BAAI/bge-small-en-v1.5`, a retrieval model, instead of mean-pooled `distilbert-base-uncased`, which was never trained for similarity.
- Scores are cosine similarities on normalised vectors and BM25 values, reported as they are, instead of the old `100 - L2 distance` percentage.
- One SQLite file holds documents, pages, chunks, the keyword index, vectors, history and the query log, replacing three separately written files.
- Deleting a document deletes its rows; it no longer re-embeds the whole corpus.
- The server listens on port 8080 (was 5000) and all endpoints moved under `/api/v1`.
- Settings are per request (strategy, models, reranking, fusion) or environment variables; the old `app_config.json` is gone.

### Removed

- The Flask application, FAISS and pickle storage, PyPDF2 (replaced by PDFium), and the old templates and screenshots.

## [2.0.0] - 2025-12-01

### Added

- Google-style search interface with a centred search box.
- Configuration from the UI: Hugging Face model id, chunk size and overlap, result count, top-k, embedding dimension, stored in `app_config.json`.
- Page numbers for PDF chunks, a "View Source" button, sorting by relevance or upload date.
- Index rebuild when the model or chunking settings change.
- Version display in the UI.

### Changed

- Search-as-you-type after three characters with a 500 ms delay.
- Uploads organised in date folders with timestamped file names.
- `document_processor.py` switched to `AutoTokenizer` and `AutoModel` so any Hugging Face model id could be used.

## [1.0.0] - 2025-11-15

### Added

- Upload of PDF, DOCX and TXT files with drag and drop.
- Semantic search with DistilBERT embeddings and a FAISS index.
- Document list, document view and delete.
- Search-as-you-type, document and chunk counts, toast notifications, a responsive layout.
