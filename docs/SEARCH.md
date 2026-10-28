# Search strategies, scoring and models

This is the reference for what happens between a query and a ranked list, and for the numbers attached to every result. It also says how to choose between the options, and how to measure instead of guess.

<p align="center"><img src="images/search-pipeline.png" alt="A search request: the query goes to keyword and semantic retrieval in parallel, the lists are fused, an optional cross-encoder reranks the top candidates, and the results are assembled into hits with their evidence" width="900" /></p>

## The three strategies

| Strategy | What it does | Strong at | Weak at |
|---|---|---|---|
| `lexical` | BM25 over the FTS5 index. Words are stemmed, so `returns` matches `returned`. Quoted phrases must appear verbatim. Common English words are dropped from the query (they stay in the index). | Exact terms: names, product codes, error messages, numbers, anything the user knows the wording of | Paraphrase, synonyms, questions in different words than the document |
| `dense` | The query is embedded with the same model as the chunks and compared by cosine similarity. | Meaning: "how do I get my money back" finds the refund policy | Rare tokens, identifiers, exact numbers; scores are not comparable across models |
| `hybrid` | Both, fused. The default. | Most queries: precise when the words match, forgiving when they do not | Costs both lookups (still tens of milliseconds) |

The keyword stage tries all terms with AND first. If that returns fewer candidates than asked for it runs the OR form and appends the extra rows, so precise matches always lead and partial matches fill the list. The parsed query (phrases, kept terms, dropped words) is returned in the response under `retrievers[].details.query` and shown in the UI.

## Fusion

Each retriever returns a ranked list of `candidates` chunks (50 by default). Two ways to merge them:

**Reciprocal rank fusion (`rrf`, default).** Each list gives a chunk `1 / (k + rank)` with `k = 60`; the sums are the fused scores. It ignores the raw scores on purpose: BM25 values and cosine similarities live on different scales, and RRF needs no calibration. A chunk on both lists reliably outranks one on a single list, which matches the intuition that agreement between two different views is the strongest signal hybrid search has. Cormack, Clarke and Buettcher, SIGIR 2009.

**Weighted (`weighted`).** Each list's scores are min-max normalised to 0..1 over that list, dense lists are averaged, and the result is `alpha * dense + (1 - alpha) * lexical` with `alpha` between 0 and 1 (`hybrid_alpha`, default 0.5). Use it when you want to lean one way, for example `alpha = 0.3` for a corpus full of part numbers. It is more sensitive to how scores are distributed in a given result set than RRF.

With more than one dense model, every model is its own list. Fusing two models is a cheap way to smooth out the blind spots of either one.

## Reranking

A cross-encoder reads the query and a candidate passage together and outputs one relevance score. It is far more precise than comparing two independent embeddings, and far more expensive: one forward pass per candidate. So it runs after fusion, only on the top `rerank_candidates` (30), and only when asked (`CSE_RERANK=true` or `rerank=true` per request; the UI has a toggle).

The default reranker is `cross-encoder/ms-marco-MiniLM-L-6-v2` (English, about 90 MB, roughly 150 to 400 ms for 30 candidates on a laptop CPU, a few tens of milliseconds on a GPU). `BAAI/bge-reranker-base` and `BAAI/bge-reranker-v2-m3` are stronger and multilingual and cost several times more. The reranker's logit is passed through a sigmoid, so its score is a probability-like number in 0..1 and becomes the confidence of the hit.

Reranked results keep both ranks in the evidence (`rank_before`, `rank_after`), so you can see what the reranker changed.

## Confidence

Every hit has `confidence` in 0..1 and a label: `strong` at or above `confidence_strong` (0.70), `likely` at or above `confidence_likely` (0.40), `weak` below. It is a comparability score, not a calibrated probability, except when the reranker is on.

| Situation | Confidence |
|---|---|
| reranker on | the reranker's sigmoid score |
| found by keyword and semantic search | `0.5 * coverage + 0.5 * similarity + 0.1 * min(coverage, similarity)` |
| keyword only | `0.85 * coverage` |
| semantic only | `0.85 * similarity` |

`coverage` is the fraction of query terms and phrases that occur in the chunk (stemmed forms count, so `refunds` covers `refund`). `similarity` maps the best cosine across models onto 0..1 between `confidence_dense_low` (0.35) and `confidence_dense_high` (0.80). The 15 percent discount for single-retriever hits reflects that agreement is the better signal. The pieces are returned as `evidence.confidence_parts`.

### Calibrating for your model

Cosine ranges differ between model families. The defaults fit `bge` and `all-MiniLM`, where an unrelated passage scores around 0.3 to 0.5 and a good match 0.7 to 0.85. The `e5` family clusters higher (unrelated 0.75, good 0.85 and up). To calibrate: run a handful of queries with `strategy=dense`, look at `evidence.dense.<model>.cosine` for hits you consider good and bad, and set `CSE_CONFIDENCE_DENSE_LOW` a little above the bad ones and `CSE_CONFIDENCE_DENSE_HIGH` around the good ones. Or turn reranking on and let the cross-encoder decide.

## Snippets and highlights

The snippet is a window of `snippet_chars` (320) around the densest cluster of matches in the chunk, snapped to word boundaries, with `...` where it was cut. `highlights` lists `[start, end]` pairs inside the snippet. For keyword hits the highlights come from FTS5 itself, so they reflect exactly what matched, stems included; for semantic-only hits the query terms are matched by prefix. Line breaks in the source become spaces so a snippet reads as one paragraph; the full chunk text is in `text` and the page text is one request away.

## Grouping

By default chunks from the same document collapse into one result: the best chunk leads and up to three other matching passages are listed under `also_matched` with their own page, span and snippet. `top_k` then counts documents. Send `group=false` to get raw chunks, which is what you want when feeding passages to another system.

## Choosing an embedding model

| Model | Dim | Size | Notes |
|---|---|---|---|
| `BAAI/bge-small-en-v1.5` (default) | 384 | 130 MB | Fast on CPU, strong on English retrieval benchmarks. Uses a query instruction, applied automatically |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | 90 MB | The fastest common choice; a good second opinion in Compare |
| `BAAI/bge-base-en-v1.5` | 768 | 440 MB | About 3x slower than small, a little more accurate |
| `intfloat/e5-base-v2` | 768 | 440 MB | Needs `query:` and `passage:` prefixes, applied automatically |
| `intfloat/multilingual-e5-small` | 384 | 470 MB | 100 languages |
| `BAAI/bge-m3` | 1024 | 2.2 GB | Multilingual, 8k-token passages. Heavy on CPU |
| `Qwen/Qwen3-Embedding-0.6B` | 1024 | 1.2 GB | Multilingual, instruction aware; comfortable on a GPU |

Any model that sentence-transformers can load works: set `CSE_EMBEDDING_MODELS` or build it from the Activity page. The dimension is read from the model. Known families get the prefixes their authors recommend; for anything else set `CSE_EMBEDDING_QUERY_PREFIX` and `CSE_EMBEDDING_DOCUMENT_PREFIX` if the model card asks for them. Changing a model does not change the chunks, so building a second index is just one embedding pass over the stored chunks, and both indexes stay searchable.

Rules of thumb: start with the default; add MiniLM to compare; move to a base or multilingual model only if the evaluation says so; turn reranking on before switching to a larger embedding model, it usually helps more.

## Comparing in the UI

The Compare page runs one query under up to four configurations and marks the documents every variant returned. The paraphrase below has almost no keyword overlap with the refund policy, which is why the keyword-only column rates it 51 percent, hybrid 74 percent and hybrid with reranking 96 percent, while the reranker pushes the unrelated documents to near zero.

<p align="center"><img src="images/compare.jpg" alt="Compare page with keyword, hybrid and reranked columns" width="900" /></p>

## Measuring instead of guessing

`contextsearch eval` scores strategies on your own queries:

```bash
contextsearch eval labels.json --strategies lexical,dense,hybrid,hybrid:weighted,hybrid+rerank,dense:intfloat/e5-base-v2 --k 10
```

On the bundled `examples/` corpus (five documents, ten queries, laptop CPU) the output looks like this:

```
strategy       recall@k   mrr@k   ndcg@k   latency
-----------------------------------------------------
lexical           1.000   0.933    0.950      2.3 ms
dense             1.000   0.950    0.963     42.3 ms
hybrid            1.000   0.950    0.963     33.5 ms
hybrid+rerank     1.000   1.000    1.000    197.1 ms
```

A corpus that small is easy for every strategy; the differences show up on real collections with hundreds of documents and queries phrased by people who have not read them. The labels file lists queries and the documents (optionally `#page`) that should come back, see [../examples/labels.json](../examples/labels.json). Twenty to fifty real queries from real users, labelled by someone who knows the corpus, tell you more than any public benchmark. Recall tells you whether the right document is in the list at all; MRR and nDCG tell you how high it sits. The Compare page in the UI is the quick visual version of the same question.

## Requests in full

Parameters accepted by `GET /api/v1/search` and `POST /api/v1/search` (see [API.md](API.md)): `q` or `query`, `strategy`, `models`, `top_k`, `candidates`, `fusion`, `alpha`, `rrf_k`, `rerank`, `rerank_model`, `group`, `prefix_last` (treat the last word as a prefix, for search-as-you-type), `source`, `ext`, `document` (search inside one document), `min_confidence` (drop weaker hits), `text` (include full chunk text) and `explain` (include evidence and trace). Everything not sent falls back to the server settings.
