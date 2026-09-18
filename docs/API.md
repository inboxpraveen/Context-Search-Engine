# API reference

Everything lives under `/api/v1`. The server also serves an interactive version of this reference at `/docs` (Swagger UI) and `/redoc`, generated from the code, and the raw schema at `/openapi.json`. The test suite checks that every path in the schema appears on this page.

## Conventions

**Authentication.** When `CSE_API_KEY` is set, every request needs it as `X-API-Key: <key>` or `Authorization: Bearer <key>`. For links a browser opens directly, such as the file endpoint, `?api_key=<key>` is accepted too; be aware that it then lands in browser history and access logs. `GET /api/v1/health` answers without a key, in a short form.

**Errors.** Always JSON, always the same shape:

```json
{"error": "document nope not found", "code": "not_found", "status": 404, "details": {}, "request_id": "0651accbceca4638"}
```

| Status | Code | Meaning |
|---|---|---|
| 400 | `validation_error` | a value is out of range or the query is empty |
| 401 | `unauthorized` | missing or wrong API key |
| 403 | `forbidden` | folder outside `CSE_ALLOWED_ROOTS`, or the action is disabled in this profile |
| 404 | `not_found` | unknown document, source, job or model |
| 409 | `conflict` | folder already a source, deleting a watched-folder document, a finished job cancelled |
| 415 | `unsupported_file` | no reader for that extension |
| 422 | `validation_error` | request body does not match the schema (`details.problems` lists the fields) or a file cannot be parsed |
| 503 | `model_error` | an embedding or reranking model could not be loaded |

**Request ids.** Every response carries `X-Request-ID`. Send your own `X-Request-ID` to have it echoed. The same id is in the server log line for the request and in `trace.request_id` of a search response.

**Timestamps** are Unix seconds. **Sizes** are bytes. Paths of documents in watched folders are returned as `path` (absolute) and `relpath` (relative to the source); uploaded documents only expose `relpath`.

## Endpoints

| Area | Method | Path | Summary |
|---|---|---|---|
| search | `GET` | `/api/v1/search` | Search with query parameters |
| search | `POST` | `/api/v1/search` | Search with a JSON body |
| search | `POST` | `/api/v1/search/compare` | Run one query under several configurations |
| search | `GET` | `/api/v1/search/recent` | Recent searches from the query log |
| sources | `GET` | `/api/v1/sources` | List sources |
| sources | `POST` | `/api/v1/sources` | Add a folder and start indexing it |
| sources | `GET` | `/api/v1/sources/{source_id}` | One source with its last scan summary |
| sources | `PATCH` | `/api/v1/sources/{source_id}` | Enable or pause a source |
| sources | `DELETE` | `/api/v1/sources/{source_id}` | Remove a source and everything indexed from it |
| sources | `POST` | `/api/v1/sources/{source_id}/sync` | Scan the folder now |
| sources | `GET` | `/api/v1/sources/{source_id}/events` | What happened to files in this source |
| documents | `GET` | `/api/v1/documents` | List documents |
| documents | `POST` | `/api/v1/documents/upload` | Upload files into the uploads source |
| documents | `GET` | `/api/v1/documents/{document_id}` | Document details, pages and revision history |
| documents | `DELETE` | `/api/v1/documents/{document_id}` | Delete an uploaded document |
| documents | `GET` | `/api/v1/documents/{document_id}/pages/{page}` | Text of one page with chunk boundaries |
| documents | `GET` | `/api/v1/documents/{document_id}/file` | Download or open the original file |
| jobs | `GET` | `/api/v1/jobs` | Recent background jobs |
| jobs | `GET` | `/api/v1/jobs/{job_id}` | One job with progress |
| jobs | `POST` | `/api/v1/jobs/{job_id}/cancel` | Ask a running job to stop |
| models | `GET` | `/api/v1/models` | Indexed embedding models, presets and reranker options |
| models | `POST` | `/api/v1/models` | Build a dense index for a model |
| models | `DELETE` | `/api/v1/models/{name}` | Drop a model's dense index |
| models | `POST` | `/api/v1/reindex` | Re-chunk documents with the current settings |
| system | `GET` | `/api/v1/health` | Liveness and component status |
| system | `GET` | `/api/v1/system/stats` | Corpus, index and query statistics |
| system | `GET` | `/api/v1/system/config` | Effective settings, secrets masked |
| system | `GET` | `/api/v1/system/metrics` | Prometheus text metrics |
| system | `GET` | `/api/v1/system/events` | Recent index events across all sources |

## Search

### `GET /api/v1/search`

| Parameter | Default | Notes |
|---|---|---|
| `q` | required | the query; quote a phrase to require it verbatim |
| `strategy` | server default (`hybrid`) | `lexical`, `dense` or `hybrid` |
| `models` | all indexed | comma-separated embedding models for dense retrieval |
| `top_k` | 10 | results to return, up to `max_top_k` |
| `candidates` | 50 | candidates per retriever before fusion |
| `fusion` | `rrf` | `rrf` or `weighted` |
| `alpha` | 0.5 | dense weight for weighted fusion, 0 to 1 |
| `rrf_k` | 60 | the k in 1/(k + rank) |
| `rerank` | server default (off) | rerank the fused candidates with the cross-encoder |
| `rerank_model` | server default | any cross-encoder model id |
| `group` | true | group chunks by document |
| `prefix_last` | false | treat the last word as a prefix; use for search-as-you-type |
| `source` | | restrict to one source id |
| `ext` | | restrict to one extension, for example `pdf` |
| `document` | | restrict to one document id |
| `min_confidence` | 0 | drop results below this confidence |
| `text` | true | include the full chunk text in each result |
| `explain` | true | include evidence, retriever details and the trace |

`POST /api/v1/search` accepts the same fields as JSON (`query` instead of `q`, `group_by_document` instead of `group`, `source_id`, `document_id`, `include_text`).

```bash
curl -s "http://localhost:8080/api/v1/search?q=refund+for+a+late+delivery&strategy=hybrid&top_k=3&text=false" \
     -H "X-API-Key: change-me"
```

```json
{
  "query": "refund for a late delivery",
  "strategy": "hybrid",
  "models": ["BAAI/bge-small-en-v1.5"],
  "fusion": "rrf",
  "rerank": false,
  "top_k": 3,
  "grouped": true,
  "total_candidates": 25,
  "results": [
    {
      "rank": 1,
      "score": 0.032787,
      "confidence": 1.0,
      "confidence_label": "strong",
      "document": {"id": "5b1f...", "filename": "refund-and-returns-policy.md", "relpath": "policies/refund-and-returns-policy.md", "title": "Refund and returns policy", "ext": "md", "source_id": "523f...", "revision": 2, "pages": 1, "indexed_at": 1789702900.1},
      "page": 1,
      "chunk_id": 14,
      "chunk_index": 0,
      "char_start": 0,
      "char_end": 1154,
      "snippet": "... support@harborvale.example with the order number.  ## Late deliveries  If an order arrives more than seven days after the promised delivery date, the customer may request a full refund ...",
      "highlights": [[57, 61], [62, 72], [128, 136], [178, 184]],
      "also_matched": [
        {"chunk_id": 15, "page": 1, "chunk_index": 1, "char_start": 1030, "char_end": 1649, "score": 0.0159, "confidence": 0.41, "snippet": "...", "highlights": [[12, 22]], "found_by": ["lexical"]}
      ],
      "evidence": {
        "found_by": ["lexical", "dense:BAAI/bge-small-en-v1.5"],
        "agreement": true,
        "fused_score": 0.032787,
        "fusion": "rrf",
        "lexical": {"rank": 1, "bm25": 6.509, "matched_terms": ["refund", "delivery", "refunds", "late", "deliveries"], "covered_terms": ["refund", "late", "delivery"], "query_terms": 3},
        "dense": {"BAAI/bge-small-en-v1.5": {"rank": 1, "cosine": 0.7431}},
        "rerank": null,
        "confidence_parts": {"lexical_coverage": 1.0, "dense_similarity": 0.874, "source": "lexical+dense"},
        "locator": {"page": 1, "char_start": 0, "char_end": 1154, "chunk_index": 0}
      }
    }
  ],
  "retrievers": [
    {"name": "lexical", "kind": "lexical", "hits": 9, "duration_ms": 2.1, "details": {"query": {"phrases": [], "terms": ["refund", "late", "delivery"], "dropped": ["for", "a"]}, "expression": "\"refund\" AND \"late\" AND \"delivery\"", "and_matches": 2, "or_fallback": true, "or_matches": 7}},
    {"name": "dense:BAAI/bge-small-en-v1.5", "kind": "dense", "hits": 25, "duration_ms": 14.9, "details": {"model": "BAAI/bge-small-en-v1.5", "vectors_searched": 26, "query_embedding_cached": false, "embed_ms": 14.6}}
  ],
  "timings": {"lexical": 2.1, "dense:BAAI/bge-small-en-v1.5": 14.9, "fusion": 0.1, "assemble": 3.2, "total": 20.8},
  "warnings": [],
  "request_id": "13031e176964400d",
  "trace": {"request_id": "13031e176964400d", "total_ms": 20.8, "spans": [{"name": "lexical", "duration_ms": 2.1, "attributes": {"hits": 9}}, "..."]}
}
```

`score` is the fused score (RRF sum or weighted blend), the raw BM25 value for `lexical` only, the cosine for `dense` only, or the reranker score when reranking is on. Use `confidence` to compare hits; use the evidence to see why.

### `POST /api/v1/search/compare`

```bash
curl -s -X POST http://localhost:8080/api/v1/search/compare -H "X-API-Key: change-me" -H "Content-Type: application/json" -d '{
  "query": "how do I get my money back",
  "variants": [
    {"label": "keywords", "strategy": "lexical"},
    {"strategy": "hybrid"},
    {"strategy": "hybrid", "rerank": true}
  ]
}'
```

Returns `variants` (each a full search response plus its `label`) and `common_documents`, the document ids every variant returned. Up to four variants per call.

### `GET /api/v1/search/recent`

`?limit=50`. Returns `queries` (query, strategy, models, rerank, result count, latency, timings, when) and `stats` (count, average and maximum latency). Empty when `CSE_QUERY_LOG=false`.

## Sources

```bash
# add a folder; it is scanned right away and watched from then on
curl -s -X POST http://localhost:8080/api/v1/sources -H "X-API-Key: change-me" -H "Content-Type: application/json" \
     -d '{"path": "/srv/docs", "name": "Shared drive", "recursive": true, "include": "", "exclude": "**/archive/**"}'
# -> 201 {"source": {...}, "job": {"id": "90d8...", "kind": "sync", "status": "queued", ...}}

curl -s -X POST http://localhost:8080/api/v1/sources/523f.../sync -H "X-API-Key: change-me"     # 202 {"job": {...}}
curl -s -X PATCH http://localhost:8080/api/v1/sources/523f... -H "X-API-Key: change-me" -H "Content-Type: application/json" -d '{"enabled": false}'
curl -s -X DELETE http://localhost:8080/api/v1/sources/523f... -H "X-API-Key: change-me"
```

A source carries `document_count`, `chunk_count`, `last_scan_at` and `last_scan` (seen, new, changed, removed, unchanged, failed, too_large, seconds). `include` overrides the server's extension list for this folder; `exclude` adds glob patterns. The uploads source is created automatically and cannot be removed. Adding a folder inside an existing source, or one that contains an existing source, is a 409.

## Documents

```bash
curl -s "http://localhost:8080/api/v1/documents?q=refund&status=indexed&sort=indexed_at&order=desc&limit=50&offset=0" -H "X-API-Key: change-me"
```

Filters: `q` (name, title or path contains), `source`, `status` (`indexed`, `failed`, `removed`, `pending`, `all`), `ext`, `sort` (`indexed_at`, `updated_at`, `filename`, `size`, `pages`, `chunks`, `revision`), `order`, `limit`, `offset`. Returns `documents`, `total`, `limit`, `offset`.

`GET /api/v1/documents/{id}` adds `source`, `page_list` (page number, characters, whether OCR was used), `revisions` and `file_available`.

`GET /api/v1/documents/{id}/pages/{page}` returns `text`, `ocr`, `pages` and `chunks` (chunk id, index, `char_start`, `char_end`) so a client can highlight the span a search result points at.

`GET /api/v1/documents/{id}/file` streams the original file inline; add `?download=true` for an attachment. Files are located by document id only.

`DELETE /api/v1/documents/{id}` removes an uploaded document and its file. Documents in a watched folder answer 409: delete or move the file on disk instead, or add an exclude pattern to the source.

### Upload

```bash
curl -s -F "files=@handbook.pdf" -F "files=@notes.md" "http://localhost:8080/api/v1/documents/upload?wait=true" -H "X-API-Key: change-me"
```

```json
{"documents": [{"id": "9c0e...", "filename": "handbook.pdf", "status": "indexed", "pages": 12, "chunks": 31, ...}],
 "jobs": [{"id": "1af2...", "kind": "file", "status": "done", ...}],
 "errors": [{"file": "notes.exe", "error": "file type .exe is not indexed; allowed: csv, docx, htm, html, log, markdown, md, pdf, pptx, rst, txt"}]}
```

The response is 201 whenever the request itself was accepted; per-file problems are listed in `errors`. With `wait=true` (the default) the call returns after indexing, up to two minutes per file; with `wait=false` it returns the queued jobs and `documents` may be empty until they finish. Files above `CSE_MAX_UPLOAD_MB` are rejected.

## Jobs

`GET /api/v1/jobs?limit=50&status=running` lists recent jobs, newest first, plus `queued`, the number of tasks waiting. A job has `kind` (`sync`, `file`, `build_model`, `reindex`), `status` (`queued`, `running`, `done`, `failed`, `cancelled`), `progress` (`total`, `done`, `current` and per-outcome counts for scans), `errors` (last 50, with `error_count`), `result` and `meta`. `POST /api/v1/jobs/{id}/cancel` asks a running job to stop after the current file.

## Models

`GET /api/v1/models` returns `dense_available` (and `reason` when it is not), the `device`, `indexed` (name, dimension, vector count, missing vectors, whether the index is loaded, whether a build is running), `configured` (from `CSE_EMBEDDING_MODELS`), `presets` with notes and sizes, and the reranker options with the server default.

`POST /api/v1/models {"name": "intfloat/e5-base-v2"}` builds a dense index for another model: every stored chunk is embedded once, in the background (202 with the job). Existing indexes keep serving searches meanwhile. `DELETE /api/v1/models/{name}` drops the vectors.

`POST /api/v1/reindex {"force": false}` re-chunks documents whose chunks were cut with different settings than the current ones; `force: true` re-chunks everything.

## System

`GET /api/v1/health` without a key: `{"status": "ok", "version": "3.0.0"}`. With a key it adds `profile`, `uptime_s`, `database`, `dense` (available, reason, device), `ocr`, `models` (name, vectors, loaded), `watcher` (enabled, running, watched source ids), `worker` (running, queued, current job) and `readers` (supported extensions).

`GET /api/v1/system/stats`: documents (indexed, failed, removed), pages, chunks, characters, database size, documents by type, models with vector counts, in-memory index sizes, memory in MB, query cache hits and misses, query statistics, uptime, queued jobs.

`GET /api/v1/system/config`: every setting with secrets masked, the chunking signature, supported readers, version.

`GET /api/v1/system/metrics`: Prometheus text format. Counters `contextsearch_http_requests_total{method,status}`, `contextsearch_searches_total{strategy}`, `contextsearch_documents_indexed_total{outcome}`, `contextsearch_jobs_total{kind}`; histograms `contextsearch_http_latency_ms`, `contextsearch_search_latency_ms{strategy}`, `contextsearch_ingest_seconds`; gauges `contextsearch_documents`, `contextsearch_chunks`, `contextsearch_jobs_queued`, `contextsearch_vector_indexes`.

`GET /api/v1/system/events?limit=100&kinds=added,updated,removed,failed,scan`: the index event log across all sources.
