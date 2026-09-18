# Configuration

Every setting is read from an environment variable with the `CSE_` prefix, or from a `.env` file in the folder you start the server from.
Profiles set sensible groups of defaults; anything you set explicitly wins over the profile.

This file is generated from `contextsearch/config.py` with `contextsearch config --markdown > docs/CONFIGURATION.md`. Edit the field help there, not here.

## Profiles

| Profile | What changes |
|---|---|
| `development` (default) | text logs on the console, binds to `127.0.0.1`, any folder can be added as a source |
| `production` | json logs written to `<data_dir>/logs/contextsearch.log`, binds to `0.0.0.0`, folder sources must sit inside `CSE_ALLOWED_ROOTS` |
| `testing` | what the test suite uses: no watcher, no periodic scans, no query log, no embedding models |

Select one with `CSE_PROFILE=production` or `contextsearch serve --profile production`.

## Settings

| Setting | Environment variable | Type | Default | What it does |
|---|---|---|---|---|
| `profile` | `CSE_PROFILE` | string | `development` | Configuration profile: development, production or testing. |
| `data_dir` | `CSE_DATA_DIR` | string | `data` | Root folder for the database, uploads and logs. |
| `log_level` | `CSE_LOG_LEVEL` | string | `INFO` | Log level: DEBUG, INFO, WARNING or ERROR. |
| `log_format` | `CSE_LOG_FORMAT` | string | `text` | Log line format: text for people, json for log shippers. The production profile uses json. |
| `log_file` | `CSE_LOG_FILE` | string |  | Also write logs to this file. The production profile defaults to <data_dir>/logs/contextsearch.log; other profiles log to the console only. |
| `host` | `CSE_HOST` | string | `127.0.0.1` | Interface the server binds to. Use 0.0.0.0 to accept connections from other machines. |
| `port` | `CSE_PORT` | integer | `8080` | Port the server listens on. |
| `api_key` | `CSE_API_KEY` | string |  | When set, every /api request must send it as X-API-Key or Authorization: Bearer. Leave empty to run without auth. |
| `cors_origins` | `CSE_CORS_ORIGINS` | string |  | Comma-separated origins allowed to call the API from a browser on another host. Empty disables CORS headers. |
| `allowed_roots` | `CSE_ALLOWED_ROOTS` | string |  | Comma-separated folders that may be added as sources. Every source path has to sit inside one of them. Empty allows any path in development and refuses all folder sources in production. |
| `max_upload_mb` | `CSE_MAX_UPLOAD_MB` | integer | `50` | Largest file accepted by the upload endpoint, in MB. |
| `include_extensions` | `CSE_INCLUDE_EXTENSIONS` | string | `pdf,docx,pptx,txt,md,markdown,rst,html,htm,csv,log` | File extensions that get indexed from watched folders and uploads. Image types are added automatically when OCR is enabled. |
| `exclude_globs` | `CSE_EXCLUDE_GLOBS` | string | `**/.git/**,**/node_modules/**,**/__pycache__/**,**/~$*,**/.~lock.*,**/.DS_Store` | Comma-separated glob patterns that are skipped while scanning. Office lock files and VCS folders are skipped by default. |
| `max_file_mb` | `CSE_MAX_FILE_MB` | integer | `200` | Files larger than this are skipped and reported, not read. |
| `scan_interval_seconds` | `CSE_SCAN_INTERVAL_SECONDS` | integer | `300` | How often every enabled folder source is rescanned for changes. 0 turns periodic scans off; file system events still trigger indexing. |
| `watch_folders` | `CSE_WATCH_FOLDERS` | boolean | `True` | Subscribe to file system events (inotify on Linux, ReadDirectoryChangesW on Windows) so changes are picked up within seconds. |
| `watch_debounce_seconds` | `CSE_WATCH_DEBOUNCE_SECONDS` | number | `2.0` | Quiet period after the last event on a file before it is indexed. Editors and copies write in bursts. |
| `ocr` | `CSE_OCR` | boolean | `False` | Run OCR on images and on PDF pages that have no text layer. Needs the ocr extra. |
| `ocr_min_page_chars` | `CSE_OCR_MIN_PAGE_CHARS` | integer | `20` | A PDF page with fewer extracted characters than this is treated as scanned and sent to OCR when OCR is on. |
| `ocr_render_scale` | `CSE_OCR_RENDER_SCALE` | number | `2.0` | Zoom factor used to render a PDF page to an image before OCR. 2.0 is about 144 dpi. |
| `chunker` | `CSE_CHUNKER` | string | `recursive` | Chunking strategy: recursive splits on paragraphs and sentences, fixed cuts every N words. A custom chunker registered in code is named here too. Part of the chunking signature: changing it means existing chunks were cut differently, and `contextsearch reindex` rebuilds them. |
| `chunk_words` | `CSE_CHUNK_WORDS` | integer | `200` | Target chunk length in words. Around 200 words fits comfortably inside a 512-token embedding window. Part of the chunking signature: changing it means existing chunks were cut differently, and `contextsearch reindex` rebuilds them. |
| `chunk_overlap_words` | `CSE_CHUNK_OVERLAP_WORDS` | integer | `30` | Words carried over from the end of one chunk into the start of the next, so sentences at a boundary are searchable in both. Part of the chunking signature: changing it means existing chunks were cut differently, and `contextsearch reindex` rebuilds them. |
| `chunk_min_words` | `CSE_CHUNK_MIN_WORDS` | integer | `40` | Trailing pieces shorter than this are merged into the previous chunk. Part of the chunking signature: changing it means existing chunks were cut differently, and `contextsearch reindex` rebuilds them. |
| `embedding_models` | `CSE_EMBEDDING_MODELS` | string | `BAAI/bge-small-en-v1.5` | Comma-separated sentence-transformers models to keep dense indexes for. Empty runs keyword search only. Each model costs one embedding pass per chunk. |
| `embedding_device` | `CSE_EMBEDDING_DEVICE` | string |  | Device for embedding and reranking models: cpu, cuda, cuda:1, mps. Empty picks cuda when available, otherwise cpu. |
| `embedding_batch_size` | `CSE_EMBEDDING_BATCH_SIZE` | integer | `32` | Chunks encoded per forward pass. Raise it on a GPU, lower it if memory is tight. |
| `embedding_max_seq_length` | `CSE_EMBEDDING_MAX_SEQ_LENGTH` | integer | `0` | Override the model's token limit. 0 keeps the model default. |
| `embedding_query_prefix` | `CSE_EMBEDDING_QUERY_PREFIX` | string |  | Text prepended to queries for models without a built-in preset. Known families (bge, e5, nomic) get the right prefix automatically. |
| `embedding_document_prefix` | `CSE_EMBEDDING_DOCUMENT_PREFIX` | string |  | Text prepended to chunks for models without a built-in preset. |
| `embedding_cache_dir` | `CSE_EMBEDDING_CACHE_DIR` | string |  | Where model weights are downloaded to. Empty uses the Hugging Face default (~/.cache/huggingface). |
| `query_cache_size` | `CSE_QUERY_CACHE_SIZE` | integer | `512` | Query embeddings kept in memory so repeated searches skip the model. |
| `default_strategy` | `CSE_DEFAULT_STRATEGY` | string | `hybrid` | Strategy used when a request does not name one: lexical, dense or hybrid. |
| `default_top_k` | `CSE_DEFAULT_TOP_K` | integer | `10` | Results returned when a request does not say. |
| `max_top_k` | `CSE_MAX_TOP_K` | integer | `100` | Upper bound a client may ask for. |
| `candidates` | `CSE_CANDIDATES` | integer | `50` | Candidates each retriever contributes before fusion. Higher gives better recall at the cost of a little latency. |
| `fusion` | `CSE_FUSION` | string | `rrf` | How hybrid combines retrievers: rrf (reciprocal rank fusion, needs no tuning) or weighted (min-max normalised scores blended with hybrid_alpha). |
| `rrf_k` | `CSE_RRF_K` | integer | `60` | The k constant in 1/(k + rank). 60 is the value from the original paper and works well in practice. |
| `hybrid_alpha` | `CSE_HYBRID_ALPHA` | number | `0.5` | Weight of the dense score in weighted fusion, between 0 (keyword only) and 1 (semantic only). |
| `rerank` | `CSE_RERANK` | boolean | `False` | Rerank fused candidates with a cross-encoder by default. Clients can turn it on per request too. |
| `rerank_model` | `CSE_RERANK_MODEL` | string | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder used for reranking. Needs the dense extra. |
| `rerank_candidates` | `CSE_RERANK_CANDIDATES` | integer | `30` | How many fused candidates are scored by the reranker. |
| `group_by_document` | `CSE_GROUP_BY_DOCUMENT` | boolean | `True` | Collapse chunks from the same document into one result that lists the other matching passages. |
| `snippet_chars` | `CSE_SNIPPET_CHARS` | integer | `320` | Length of the highlighted excerpt shown for each result. |
| `lexical_stopwords` | `CSE_LEXICAL_STOPWORDS` | boolean | `True` | Drop common English words from keyword queries. They stay in the index, so quoted phrases still match them. |
| `confidence_dense_low` | `CSE_CONFIDENCE_DENSE_LOW` | number | `0.35` | Cosine similarity that maps to 0 confidence for a semantic-only hit. Depends on the model; see docs/SEARCH.md for calibration. |
| `confidence_dense_high` | `CSE_CONFIDENCE_DENSE_HIGH` | number | `0.8` | Cosine similarity that maps to full confidence for a semantic-only hit. |
| `confidence_strong` | `CSE_CONFIDENCE_STRONG` | number | `0.7` | Confidence at or above this is labelled strong. |
| `confidence_likely` | `CSE_CONFIDENCE_LIKELY` | number | `0.4` | Confidence at or above this is labelled likely; below is weak. |
| `query_log` | `CSE_QUERY_LOG` | boolean | `True` | Record every search (query text, strategy, latency, result count) in the database for the activity view. |
| `query_log_keep` | `CSE_QUERY_LOG_KEEP` | integer | `10000` | How many recent searches to keep. Older rows are trimmed. |
| `vector_compact_ratio` | `CSE_VECTOR_COMPACT_RATIO` | number | `0.2` | The in-memory matrix is rebuilt once this fraction of its rows belong to deleted chunks. |

## Derived paths

| Path | Location |
|---|---|
| database | `<data_dir>/contextsearch.db` (plus `-wal` and `-shm` files while the server runs) |
| uploads | `<data_dir>/uploads/<YYYYMMDD>/` |
| logs | `<data_dir>/logs/contextsearch.log` in production, or wherever `log_file` points |

## Examples

```bash
# keyword search only, no model download at all
CSE_EMBEDDING_MODELS= contextsearch serve

# two dense indexes to compare, reranking on, served on the LAN
CSE_EMBEDDING_MODELS=BAAI/bge-small-en-v1.5,sentence-transformers/all-MiniLM-L6-v2 \
CSE_RERANK=true CSE_HOST=0.0.0.0 CSE_API_KEY=change-me contextsearch serve

# production: locked-down roots, json logs
CSE_PROFILE=production CSE_ALLOWED_ROOTS=/srv/docs,/mnt/share CSE_API_KEY=change-me contextsearch serve
```
