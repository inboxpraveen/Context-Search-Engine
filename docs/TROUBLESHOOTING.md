# Troubleshooting

Start with `contextsearch doctor`. It checks the Python build, SQLite FTS5, the optional extras, the model cache and the data folder, and prints one line per check. Then look at the log: text on the console in development, JSON in `<data_dir>/logs/contextsearch.log` in production. Every API error carries a `request_id` that also appears in the log line for that request.

## Install and start

**`pip install` on Linux downloads gigabytes.** The default torch wheel on Linux includes CUDA. Install the CPU build first: `pip install torch --index-url https://download.pytorch.org/whl/cpu`, then `pip install -e ".[dense]"`. Or skip the dense extra entirely for keyword search only.

**`this Python build has SQLite without FTS5`.** Rare, but some distribution builds of Python link an old or stripped SQLite. Use Python from python.org, conda, or `uv`, or install the `pysqlite3-binary` package and import it as `sqlite3` before starting.

**`address already in use`.** Something else is on port 8080. Set `CSE_PORT` or `contextsearch serve --port 9000`.

**The first search takes 10 to 30 seconds.** The embedding model is loading. The engine warms models in the background after start; searches before that wait for the load. Subsequent searches take milliseconds. In production keep the process running.

**`dense search needs the dense extra`** or `hybrid ran as keyword search` in the warnings. sentence-transformers is not installed, or no dense index has been built yet. `pip install -e ".[dense]"`, then either set `CSE_EMBEDDING_MODELS` and restart, or build a model from the Activity page.

**Model download fails (no internet, proxy).** Download the model on a connected machine (`python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"`), copy `~/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5` to the server, and set `HF_HUB_OFFLINE=1`. Set `HF_HOME` if the cache should live elsewhere (the Docker image uses `/data/models`).

**The symlink notice from huggingface_hub on Windows.** Harmless; the cache uses copies instead of links. Enabling Developer Mode in Windows makes it go away.

## Indexing

**A folder shows `folder missing` after a scan.** The path is not mounted or not readable by the service user. Nothing is removed from the index in that case; fix the mount and scan again.

**Files are skipped.** Check the extension list (`CSE_INCLUDE_EXTENSIONS`, and the source's own `include` if set), the exclude patterns (`**/.git/**` and Office lock files are excluded by default), and `CSE_MAX_FILE_MB`. The scan summary on the Sources page counts `ignored` and `too large`. Images need `CSE_OCR=true` and the OCR extra.

**A document shows `failed`.** Open it from the Documents page: the error is on the detail view and in the source's events. Common causes: a file open in Word or Acrobat on Windows (locked, retried on the next scan), an encrypted PDF, a corrupt file. Failed documents are retried on every scan, so a temporary problem fixes itself.

**A document has `no text`.** The reader returned nothing, usually a scanned PDF without a text layer or an image-only page. Set `CSE_OCR=true` with the OCR extra installed and run `contextsearch reindex --force` (or scan again after touching the file).

**Changes are not picked up.** File system events do not arrive on most network shares and in some virtual machine setups. The periodic scan catches everything within `CSE_SCAN_INTERVAL_SECONDS` (default 300). Lower it, or trigger a scan from the Sources page or `POST /api/v1/sources/{id}/sync`. Make sure the file's mtime actually changed; a tool that writes without updating mtime and size is invisible to a scan until the file is hashed for another reason.

**Editing a file re-indexes it twice.** Some editors write a temporary file and rename it, which produces a remove and an add. The result is correct; the revision counter grows by one per real change of content, not per event.

**The chunk counts changed after an upgrade or a settings change.** Chunking settings are part of a signature stored with each document. Run `contextsearch reindex` to re-cut documents chunked with old settings; the Activity page shows the current signature.

**Indexing is slow.** Embedding on CPU runs at tens of chunks per second for 200-word chunks. Keyword search is available as soon as text is written; the dense index fills in behind. A GPU, shorter chunks or a single model all help; [PERFORMANCE.md](PERFORMANCE.md) has numbers.

**`database is locked`.** Another process has the file open for writing: a second `contextsearch serve` on the same data folder, or a CLI command such as `index` running while the server runs. Use one process per data folder; the CLI search and stats commands only read and are fine alongside the server.

## Search quality

**Keyword search finds nothing for a query that should match.** Words are stemmed and common English words are dropped; the response's `retrievers[0].details.query` shows what was kept. A phrase in quotes must appear verbatim. Numbers and codes with punctuation (`v1.5`, `AB-123`) are split on the punctuation by the tokenizer; search for the parts.

**Semantic results look random.** Check `evidence.dense.<model>.cosine`. Values below 0.4 with `bge` or MiniLM models mean nothing similar exists in the corpus; the retriever always returns its top candidates, so the confidence label and `min_confidence` are what tell you they are weak. A wrong query prefix can also flatten similarities: known model families get theirs automatically, others need `CSE_EMBEDDING_QUERY_PREFIX`.

**Hybrid ranks a keyword match below a vague semantic one.** Try `fusion=weighted` with `alpha=0.3` for a keyword-heavy corpus, or turn reranking on. Compare the options on the Compare page or with `contextsearch eval` and a labels file before changing the default.

**Confidence is 100 percent for results that are not great.** The confidence blend is a heuristic; see [SEARCH.md](SEARCH.md) for the formula and how to calibrate `CSE_CONFIDENCE_DENSE_LOW` and `CSE_CONFIDENCE_DENSE_HIGH` for your model. With reranking on, the reranker's probability replaces the heuristic.

**Results from the same document crowd the list.** That is what `group=true` (the default) prevents; if you sent `group=false`, raw chunks come back. Or raise `top_k`.

## API and UI

**401 on every call.** `CSE_API_KEY` is set on the server; send it as `X-API-Key`. In the UI, open Settings (the gear) and paste the key; it is stored in the browser.

**403 when adding a folder.** In production the folder must sit inside `CSE_ALLOWED_ROOTS`. The error lists the allowed roots.

**409 when adding a folder.** It is already a source, or inside one, or contains one. Remove the overlapping source first.

**Deleting a document returns 409.** Documents in watched folders follow the folder: delete or move the file, or add an exclude pattern to the source. Only uploads can be deleted through the API.

**The UI looks old after an upgrade.** The page links its assets with the version in the URL, so a hard reload (Ctrl+Shift+R) picks up the new files.

**`Open file` gives 404.** The file is no longer at its original path (the document detail shows `file_available`). Removed documents keep their text in the index for the viewer but not the file.

## Getting help

Open an issue with the output of `contextsearch doctor`, the log lines around the request id, the version, the operating system and Python version, and what you expected instead. Search-quality questions are much easier to answer with a query, the result you got and the result you wanted.
