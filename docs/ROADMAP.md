# Roadmap

What is planned, in rough order. Nothing here is promised for a date; issues and pull requests move things up.

## Images and audio

- **Image search with CLIP-style embeddings.** The store already keeps one vector per (chunk, model), the embedder interface is one class, and OCR already turns text in images into searchable chunks. The missing pieces are an image reader that produces one chunk per image, an embedder that encodes images and text into the same space, and a viewer that shows the image instead of a page. Text queries would then find pictures by content, not only by the words on them.
- **Audio.** [AudioFP](https://github.com/inboxpraveen/Audio-Fingerprint) already does audio fingerprinting as a separate service. The plan is a bridge rather than a merge: transcripts become documents here, with timestamps as the locator instead of character offsets.

## Scale

- **Approximate vector index** behind the same `VectorIndex` interface (FAISS or hnswlib) for corpora past a few million chunks, with exact search kept as the default because it is deterministic and never misses.
- **PostgreSQL with pgvector** as an alternative `Store`, for teams that already run Postgres and want several read replicas. SQLite stays the default.
- **Read-only replicas.** Several processes serving searches from one index while a single one writes.

## Search

- **Query understanding without a language model:** spelling correction from the corpus vocabulary, automatic phrase detection for identifiers, per-field boosts (title, path).
- **Per-source and per-type weighting** in fusion.
- **Learned calibration** of the confidence score from the evaluation labels, so the label reflects measured precision on your corpus.
- **Highlights inside PDF pages** in the viewer, using PDFium's character boxes, so the exact span is marked on the rendered page as well as in the text.

## Operations

- **Retention** for the query log and events, and an export.
- **Prometheus and OpenTelemetry**: the trace object already records spans; exporting them over OTLP is a small step.
- **Multi-user** access with per-source permissions, once someone needs it.

## Not planned

- Answer generation with a language model. The engine is for finding passages and showing why; what to do with them is up to the application on top.
- Cloud-only dependencies. Everything must keep working on a laptop with no internet after the first model download.
