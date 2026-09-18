"""Command line: serve the API, index folders, search, check the install.

    contextsearch serve [--host H] [--port P] [--profile production]
    contextsearch index PATH [--name N] [--no-recursive]
    contextsearch sync [SOURCE_ID]
    contextsearch search "query" [--strategy hybrid] [--model M] [--rerank] [--top-k 10] [--json]
    contextsearch sources | documents | models | stats | jobs
    contextsearch build MODEL | drop MODEL | reindex [--force]
    contextsearch eval labels.json [--strategies lexical,hybrid,hybrid+rerank] [--k 10]
    contextsearch doctor
    contextsearch config [--markdown | --json]
    contextsearch db check | vacuum

Every command accepts --data-dir, --profile and --log-level. Set CSE_* variables
or a .env file for everything else (see docs/CONFIGURATION.md).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sqlite3
import sys
from typing import Any

from . import __version__
from .config import Settings
from .errors import ContextSearchError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="contextsearch", description="Hybrid document search you can run yourself.")
    parser.add_argument("--version", action="version", version=f"contextsearch {__version__}")
    parser.add_argument("--data-dir", help="Folder for the database, uploads and logs (CSE_DATA_DIR).")
    parser.add_argument(
        "--profile", choices=("development", "production", "testing"), help="Configuration profile (CSE_PROFILE)."
    )
    parser.add_argument("--log-level", help="DEBUG, INFO, WARNING or ERROR (CSE_LOG_LEVEL).")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("serve", help="Run the API and web UI.")
    p.add_argument("--host")
    p.add_argument("--port", type=int)
    p.add_argument("--reload", action="store_true", help="Restart on code changes (development only).")

    p = sub.add_parser(
        "index", help="Add a folder as a source and scan it now (waits for the scan; use the API to queue one)."
    )
    p.add_argument("path")
    p.add_argument("--name", default="")
    p.add_argument("--no-recursive", action="store_true")
    p.add_argument(
        "--include", default="", help="Comma-separated extensions that override the default for this folder."
    )
    p.add_argument("--exclude", default="", help="Extra comma-separated glob patterns to skip.")

    p = sub.add_parser("sync", help="Rescan one source, or all of them.")
    p.add_argument("source_id", nargs="?")

    p = sub.add_parser("search", help="Search from the terminal.")
    p.add_argument("query")
    p.add_argument("--strategy", choices=("lexical", "dense", "hybrid"))
    p.add_argument("--model", action="append", help="Embedding model for dense retrieval; repeatable.")
    p.add_argument("--rerank", action="store_true")
    p.add_argument("--fusion", choices=("rrf", "weighted"))
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--source")
    p.add_argument("--ext")
    p.add_argument("--json", action="store_true", help="Print the full JSON response.")

    sub.add_parser("sources", help="List sources.")
    p = sub.add_parser("documents", help="List documents.")
    p.add_argument("--q", default=None, help="Filter by name, title or path.")
    p.add_argument("--status", default="indexed")
    p.add_argument("--limit", type=int, default=50)
    sub.add_parser("models", help="List dense indexes and model presets.")
    p = sub.add_parser("build", help="Build a dense index for a model.")
    p.add_argument("model")
    p = sub.add_parser("drop", help="Drop a model's dense index.")
    p.add_argument("model")
    p = sub.add_parser("reindex", help="Re-chunk documents with the current chunking settings.")
    p.add_argument("--force", action="store_true", help="Every document, not only those with outdated chunks.")
    sub.add_parser("stats", help="Corpus and index statistics.")
    sub.add_parser("jobs", help="Recent background jobs (only meaningful while the server runs in this process).")

    p = sub.add_parser("eval", help="Score strategies against labelled queries.")
    p.add_argument("labels", help="JSON file with queries and relevant documents.")
    p.add_argument(
        "--strategies",
        default="lexical,hybrid",
        help="Comma-separated, e.g. lexical,dense,hybrid,hybrid+rerank,hybrid:weighted",
    )
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--json", action="store_true")

    sub.add_parser("doctor", help="Check Python, SQLite, optional extras, models and the data folder.")
    p = sub.add_parser("config", help="Show the effective configuration.")
    p.add_argument("--markdown", action="store_true", help="Print the configuration reference (docs/CONFIGURATION.md).")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("db", help="Database maintenance.")
    p.add_argument("action", choices=("check", "vacuum", "optimize"))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args) or 0
    except ContextSearchError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        if exc.details:
            print(json.dumps(exc.details, indent=2, default=str), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


def _settings(args: argparse.Namespace, **extra: Any) -> Settings:
    overrides = {"data_dir": args.data_dir, "profile": args.profile, "log_level": args.log_level, **extra}
    return Settings.from_env(**{k: v for k, v in overrides.items() if v is not None})


def run(args: argparse.Namespace) -> int:
    if args.command == "config":
        if args.markdown:
            print(Settings.markdown_reference())
            return 0
        settings = _settings(args)
        if args.json:
            print(json.dumps(settings.to_dict(), indent=2))
        else:
            for key, value in settings.to_dict().items():
                print(f"{key:<28} {value}")
            print(f"{'chunking_signature':<28} {settings.chunking_signature()}")
        return 0
    if args.command == "doctor":
        return doctor(_settings(args))
    if args.command == "serve":
        return serve(args)

    from .engine import Engine

    # Read-only commands print their own output; keep the log quiet unless asked.
    quiet = args.command in ("search", "eval", "sources", "documents", "models", "stats", "jobs", "drop")
    settings = _settings(args, log_level="WARNING" if quiet and not args.log_level else None)
    if args.command == "db":
        return db_command(settings, args.action)

    engine = Engine(settings)
    needs_worker = args.command in ("index", "sync", "build", "reindex", "jobs")
    engine.start(warm=False, workers=needs_worker)
    try:
        return command(engine, args)
    finally:
        engine.close()


def serve(args: argparse.Namespace) -> int:
    settings = _settings(args, host=args.host, port=args.port)
    import uvicorn

    from .api import create_app

    if args.reload:
        os.environ.setdefault("CSE_DATA_DIR", settings.data_dir)
        uvicorn.run(
            "contextsearch.api.app:create_app",
            factory=True,
            host=settings.host,
            port=settings.port,
            reload=True,
            log_level="warning",
        )
        return 0
    app = create_app(settings=settings)
    print(
        f"contextsearch {__version__} at http://{settings.host}:{settings.port}  (API docs at /docs, profile {settings.profile})"
    )
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="warning", access_log=False)
    return 0


def command(engine: Any, args: argparse.Namespace) -> int:
    if args.command == "index":
        source, job = engine.add_folder(args.path, args.name, not args.no_recursive, args.include, args.exclude)
        print(f"added source {source.id} ({source.path})")
        return _follow(engine, job)

    if args.command == "sync":
        jobs = [engine.sync_source(args.source_id)] if args.source_id else engine.sync_all()
        code = 0
        for job in jobs:
            code |= _follow(engine, job)
        return code

    if args.command == "search":
        response = engine.search(
            args.query,
            strategy=args.strategy or "",
            models=args.model,
            rerank=True if args.rerank else None,
            fusion=args.fusion or "",
            top_k=args.top_k,
            source_id=args.source,
            ext=args.ext,
        )
        data = response.to_dict()
        if args.json:
            print(json.dumps(data, indent=2, ensure_ascii=False))
            return 0
        print(
            f"{len(data['results'])} results for {data['query']!r} ({data['strategy']}, {data['timings']['total']} ms)"
        )
        for w in data["warnings"]:
            print(f"  note: {w}")
        for r in data["results"]:
            doc = r["document"]
            print(
                f"\n#{r['rank']}  {doc['relpath']}  page {r['page']}  [{r['confidence_label']} {r['confidence']:.2f}]  found by {', '.join(r['evidence']['found_by'])}"
            )
            print("    " + r["snippet"].replace("\n", " ")[:300])
        return 0

    if args.command == "sources":
        for s in engine.store.list_sources():
            flag = "" if s.enabled else " (paused)"
            print(
                f"{s.id}  {s.kind:<8} {s.document_count:>6} docs  {s.chunk_count:>8} chunks  {s.name}{flag}  {s.path}"
            )
        return 0

    if args.command == "documents":
        docs, total = engine.store.list_documents(status=args.status, q=args.q, limit=args.limit)
        for d in docs:
            print(f"{d.id}  {d.status:<8} rev {d.revision:<3} {d.pages:>5} p {d.chunks:>6} ch  {d.relpath}")
        print(f"{len(docs)} of {total}")
        return 0

    if args.command == "models":
        info = engine.models_info()
        if info:
            for m in info:
                print(
                    f"{m['name']:<48} {m['dimension']:>5}d  {m['vectors']:>9} vectors  {m['missing_vectors']:>7} missing"
                )
        else:
            print("no dense index yet; build one with: contextsearch build BAAI/bge-small-en-v1.5")
        from .embedding import PRESETS

        print("\npresets:")
        for p in PRESETS.values():
            print(f"  {p.name:<46} {p.dimension:>5}d  {p.note}")
        return 0

    if args.command == "build":
        return _follow(engine, engine.build_dense_index(args.model))

    if args.command == "drop":
        engine.drop_dense_index(args.model)
        print(f"dropped {args.model}")
        return 0

    if args.command == "reindex":
        return _follow(engine, engine.reindex(force=args.force))

    if args.command == "stats":
        print(json.dumps(engine.stats(), indent=2, default=str))
        return 0

    if args.command == "jobs":
        for j in engine.jobs.list():
            print(f"{j.id}  {j.status:<9} {j.kind:<12} {j.label}")
        return 0

    if args.command == "eval":
        from .evaluate import StrategySpec, evaluate, format_table, load_labels

        labels = load_labels(args.labels)
        specs = [StrategySpec.parse(s) for s in args.strategies.split(",") if s.strip()]
        reports = evaluate(engine, labels, specs, args.k)
        if args.json:
            print(json.dumps([r.to_dict() for r in reports], indent=2))
        else:
            print(f"{len(labels)} queries, k={args.k}\n")
            print(format_table(reports))
            for r in reports:
                misses = [q.query for q in r.results if q.hit_rank is None]
                if misses:
                    print(
                        f"\n{r.label}: missed {len(misses)}: "
                        + "; ".join(misses[:5])
                        + (" ..." if len(misses) > 5 else "")
                    )
                for w in sorted(set(r.warnings)):
                    print(f"  note ({r.label}): {w}")
        return 0

    print(f"unknown command {args.command}", file=sys.stderr)
    return 2


def _follow(engine: Any, job: Any) -> int:
    """Print a job's progress until it finishes."""
    last = ""
    while not job.wait(0.5):
        p = job.progress
        if p.get("total"):
            line = f"\r{job.label}: {p.get('done', 0)}/{p['total']}  {p.get('current', '')[:60]:<60}"
        else:
            line = f"\r{job.label}: {job.status} ..."
        if line != last:
            sys.stdout.write(line)
            sys.stdout.flush()
            last = line
    sys.stdout.write("\r" + " " * 100 + "\r")
    summary = {k: v for k, v in job.result.items() if not isinstance(v, (dict, list))}
    print(f"{job.label}: {job.status}  {json.dumps(summary, default=str)}")
    for err in job.errors[:10]:
        print(f"  {err.get('path', '')}: {err.get('error', '')}")
    if len(job.errors) > 10:
        print(f"  ... and {len(job.errors) - 10} more; see the Activity page or the log")
    return 0 if job.status == "done" else 1


def db_command(settings: Settings, action: str) -> int:
    from .store import Store

    if not settings.db_path.exists():
        print(f"no database at {settings.db_path}")
        return 1
    store = Store(settings.db_path)
    try:
        if action == "check":
            result = store.integrity_check()
            print(f"integrity_check: {result}")
            print(json.dumps(store.stats(), indent=2, default=str))
            return 0 if result == "ok" else 1
        if action == "vacuum":
            before = settings.db_path.stat().st_size
            store.vacuum()
            print(f"vacuumed: {before:,} -> {settings.db_path.stat().st_size:,} bytes")
        if action == "optimize":
            store.optimize_fts()
            print("FTS index optimised")
        return 0
    finally:
        store.close()


# ---------------------------------------------------------------------- doctor


def doctor(settings: Settings) -> int:
    rows: list[tuple[str, str, str]] = []

    def ok(name: str, detail: str = "") -> None:
        rows.append(("ok", name, detail))

    def warn(name: str, detail: str = "") -> None:
        rows.append(("warn", name, detail))

    def fail(name: str, detail: str = "") -> None:
        rows.append(("fail", name, detail))

    py = platform.python_version()
    (ok if sys.version_info >= (3, 10) else fail)("python", f"{py} on {platform.system()} {platform.release()}")

    try:
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE VIRTUAL TABLE t USING fts5(x, tokenize='porter unicode61')")
        ok("sqlite", f"{sqlite3.sqlite_version} with FTS5")
    except sqlite3.OperationalError as exc:
        fail("sqlite", f"{sqlite3.sqlite_version}: FTS5 missing ({exc}); use Python from python.org or conda")

    from importlib.metadata import PackageNotFoundError, version

    for module, dist, label in (
        ("numpy", "numpy", "numpy"),
        ("pypdfium2", "pypdfium2", "PDF reader (pypdfium2)"),
        ("docx", "python-docx", "Word reader (python-docx)"),
        ("pptx", "python-pptx", "PowerPoint reader (python-pptx)"),
        ("watchdog", "watchdog", "file watching (watchdog)"),
        ("fastapi", "fastapi", "API (fastapi)"),
        ("uvicorn", "uvicorn", "server (uvicorn)"),
    ):
        try:
            __import__(module)
        except Exception as exc:
            fail(label, f"not importable: {exc}")
            continue
        try:
            ok(label, version(dist))
        except PackageNotFoundError:
            ok(label)

    from .embedding import PRESETS, dense_available, pick_device

    dense_ok, reason = dense_available()
    if dense_ok:
        import sentence_transformers
        import torch

        device = settings.embedding_device or pick_device()
        detail = (
            f"sentence-transformers {sentence_transformers.__version__}, torch {torch.__version__}, device {device}"
        )
        if device == "cuda":
            detail += f" ({torch.cuda.get_device_name(0)})"
        ok("dense search", detail)
    else:
        warn(
            "dense search",
            f"not available ({reason}); keyword search still works. Install with: pip install 'contextsearch[dense]'",
        )

    from .readers import ocr

    if settings.ocr:
        (ok if ocr.available() else fail)(
            "ocr",
            "RapidOCR ready" if ocr.available() else f"CSE_OCR is on but the extra is missing: {ocr.why_unavailable()}",
        )
    else:
        ok(
            "ocr",
            "off (set CSE_OCR=true to read scanned pages and images)"
            + ("" if ocr.available() else "; extra not installed"),
        )

    for model in settings.embedding_model_list:
        cached = _model_cached(model)
        preset = PRESETS.get(model)
        note = (
            f"{preset.dimension}d, about {preset.size_mb} MB"
            if preset
            else "not a preset; the dimension is read after the first load"
        )
        if cached:
            ok(f"model {model}", f"in the Hugging Face cache ({note})")
        elif dense_ok:
            warn(f"model {model}", f"not cached yet; downloaded on first use ({note})")
        else:
            warn(f"model {model}", "configured but dense search is unavailable")

    try:
        settings.ensure_dirs()
        probe = settings.data_path / ".write-test"
        probe.write_text("ok")
        probe.unlink()
        free = shutil.disk_usage(settings.data_path).free
        ok("data folder", f"{settings.data_path} writable, {free / 1e9:.1f} GB free")
    except OSError as exc:
        fail("data folder", f"{settings.data_path}: {exc}")

    if settings.db_path.exists():
        try:
            from .store import Store

            store = Store(settings.db_path)
            integrity = store.integrity_check()
            s = store.stats()
            store.close()
            (ok if integrity == "ok" else fail)(
                "database",
                f"{s['documents']} documents, {s['chunks']} chunks, {s['database_bytes'] / 1e6:.1f} MB, integrity {integrity}",
            )
        except Exception as exc:
            fail("database", str(exc))
    else:
        ok("database", "not created yet; the first run creates it")

    if settings.is_production:
        (ok if settings.api_key else warn)(
            "api key", "set" if settings.api_key else "CSE_API_KEY is empty; anyone who can reach the port can search"
        )
        (ok if settings.allowed_roots else warn)(
            "allowed roots",
            settings.allowed_roots or "CSE_ALLOWED_ROOTS is empty; folder sources cannot be added through the API",
        )

    worst = 0
    for status, name, detail in rows:
        mark = {"ok": "OK  ", "warn": "WARN", "fail": "FAIL"}[status]
        print(f"[{mark}] {name:<34} {detail}")
        worst = max(worst, {"ok": 0, "warn": 0, "fail": 1}[status])
    print(f"\ncontextsearch {__version__}, profile {settings.profile}, data dir {settings.data_path}")
    return worst


def _model_cached(model: str) -> bool:
    try:
        from huggingface_hub import scan_cache_dir
    except Exception:
        return False
    try:
        for repo in scan_cache_dir().repos:
            if repo.repo_id == model and repo.repo_type == "model":
                return True
    except Exception:
        return False
    return False


if __name__ == "__main__":
    raise SystemExit(main())
