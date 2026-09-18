# Performance and sizing

Numbers below were measured on a laptop with an Intel i7-14650HX (16 cores), 16 GB of RAM and no GPU in use, with contextsearch 3.0 on Python 3.12 and the default model `BAAI/bge-small-en-v1.5` (384 dimensions). Your machine will differ; the shapes of the curves will not.

## Memory

Every embedding model keeps its vectors in memory as float32: `chunks x dimension x 4 bytes`, plus up to the same again while the matrix grows (capacity doubles, then settles after a restart).

| Chunks | 384 dims (bge-small, MiniLM) | 768 dims (bge-base, e5-base) | 1024 dims (bge-m3) |
|---|---|---|---|
| 10,000 | 15 MB | 30 MB | 40 MB |
| 100,000 | 150 MB | 300 MB | 400 MB |
| 500,000 | 770 MB | 1.5 GB | 2 GB |
| 1,000,000 | 1.5 GB | 3 GB | 4 GB |

A 200-word chunk is about a third of a page, so 100,000 chunks is roughly 30,000 pages or a few thousand typical PDFs. Add the model itself (130 MB for bge-small, 440 MB for a base model, 90 MB for the default reranker) and about 1 GB for torch. A machine with 8 GB handles a few hundred thousand chunks with one small model comfortably.

The database on disk is dominated by vectors (the same bytes again, uncompressed) and the FTS index; page text and chunks are small. 100,000 chunks with one 384-dimension model came to about 780 MB.

## Search latency

| Stage | 10k chunks | 100k chunks | 500k chunks | 1M chunks |
|---|---|---|---|---|
| exact vector search, one model | 1 ms | 9 ms | 40 ms | 48 ms |
| the same with a source or type filter | 1.4 ms | 11 ms | 51 ms | 56 ms |
| query embedding on CPU (uncached) | 10 ms | 10 ms | 10 ms | 10 ms |
| keyword (FTS5), typical query | 1 ms | 2 to 12 ms | 10 to 60 ms | 20 to 120 ms |
| keyword, very broad OR query | 5 ms | 43 ms | (scales with matches) | |
| fusion and assembly | under 5 ms | | | |
| reranking 30 candidates, MiniLM-L6 on CPU | 150 to 400 ms | | | |

So a hybrid search over 100,000 chunks answers in 20 to 40 ms on CPU, and a few milliseconds more per extra embedding model. Repeated queries skip the embedding step (`query_cache_size`). Reranking is the only stage that costs real time on a CPU; on a GPU it drops to tens of milliseconds.

The keyword numbers depend on how common the query terms are, not on corpus size alone: FTS5 scores every chunk that matches at least one term. A query made of frequent words over a large corpus is the slow case; that is also why common English words are dropped from keyword queries by default.

## Indexing throughput

| Step | Rate on this CPU |
|---|---|
| scanning a folder tree (stat only) | thousands of files per second |
| hashing changed files | disk speed, about 1 GB/s |
| PDF text extraction (PDFium) | 50 to 200 pages per second |
| OCR of a scanned page (RapidOCR) | 2 to 3 seconds per page |
| chunking and writing to SQLite with FTS | 6,000 chunks per second |
| embedding 200-word chunks, bge-small, CPU | 30 to 40 chunks per second |
| embedding 60-word chunks, bge-small, CPU | about 115 chunks per second |
| embedding on a laptop GPU (RTX 5060 class) | roughly 20 to 50 times the CPU rate |

Embedding is the bottleneck of a first index on CPU: 100,000 chunks take about an hour. The text is committed before the vectors, so keyword search works from the first minute and the dense index fills in behind it. Every additional model costs one more pass. On a GPU the same corpus embeds in a few minutes; set `CSE_EMBEDDING_DEVICE=cuda` or let it be picked up automatically, and raise `CSE_EMBEDDING_BATCH_SIZE` to 128 or 256.

A rescan of a tree with nothing new is a stat walk: 10,000 files in about a second on local disk, longer on a network share. Only files whose size or mtime changed are hashed and read.

## Startup

Loading a vector matrix from SQLite runs at about 25,000 vectors per second (100,000 vectors in 4 seconds). Loading an embedding model takes 5 to 25 seconds the first time in a process; the engine warms models in the background right after start, so the first search after a restart may wait a few seconds for it. The reranker loads on first use unless `CSE_RERANK=true`, in which case it is warmed too.

## What to tune

| Goal | Knob |
|---|---|
| faster first index | a GPU; shorter chunks (`chunk_words`); fewer models; `embedding_batch_size` up on GPU |
| less memory | one model; a 384-dimension model over a 768 one; `vector_compact_ratio` lower if you delete a lot |
| faster searches | leave `rerank` off by default and let users toggle it; `candidates` down from 50 to 20 |
| better recall | `candidates` up to 100; a second model in hybrid; `chunk_overlap_words` up |
| better precision at the top | reranking; `min_confidence` on the request |
| less disk | `contextsearch db vacuum` after large deletions; keep uploads elsewhere |

## Beyond a few million chunks

Exact search over a float32 matrix stays under 100 ms up to a few million vectors on a modern CPU, and it never misses a result. Past that, memory rather than latency becomes the constraint. The plan for that scale (an approximate index such as FAISS or hnswlib behind the same `VectorIndex` interface, and PostgreSQL with pgvector as an alternative store) is on the [roadmap](ROADMAP.md). Until then, split very large collections into several data folders, one process each.
