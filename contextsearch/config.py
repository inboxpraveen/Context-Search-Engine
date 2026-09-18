"""Typed configuration, read from the environment.

Settings come from three layers. Each layer overrides the one before it:

1. The field defaults on :class:`Settings`.
2. The selected profile: ``development``, ``production`` or ``testing``.
3. Environment variables with the ``CSE_`` prefix. These can also come from a
   ``.env`` file in the working directory.

Example::

    CSE_PROFILE=production CSE_PORT=9000 CSE_API_KEY=secret contextsearch serve

Every option is documented in ``docs/CONFIGURATION.md``. That file is generated
from the field metadata below (``contextsearch config --markdown``), so the two
cannot drift apart.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import Any

from .errors import ConfigurationError

ENV_PREFIX = "CSE_"
PROFILES = ("development", "production", "testing")
STRATEGIES = ("lexical", "dense", "hybrid")
FUSIONS = ("rrf", "weighted")
IMAGE_EXTENSIONS = frozenset({"png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"})

# Bump this when the chunker changes how it cuts text in a way that makes old
# chunks incomparable with new ones. Parameter changes are covered by the
# signature already.
CHUNKING_ALGORITHM_VERSION = 1


def _doc(help_text: str, **extra: Any) -> dict[str, Any]:
    meta = {"help": help_text}
    meta.update(extra)
    return meta


@dataclass
class Settings:
    """All runtime configuration.

    Field metadata keys: ``help`` is the text shown in the docs, ``secret``
    marks a value that is never printed, ``chunking`` marks a parameter that
    goes into the chunking signature and ``env=False`` stops a field being set
    from the environment.
    """

    # ------------------------------------------------------------------ general
    profile: str = field(
        default="development", metadata=_doc("Configuration profile: development, production or testing.")
    )
    data_dir: str = field(default="data", metadata=_doc("Root folder for the database, uploads and logs."))
    log_level: str = field(default="INFO", metadata=_doc("Log level: DEBUG, INFO, WARNING or ERROR."))
    log_format: str = field(
        default="text",
        metadata=_doc("Log line format: text for people, json for log shippers. The production profile uses json."),
    )
    log_file: str = field(
        default="",
        metadata=_doc(
            "Also write logs to this file. The production profile defaults to <data_dir>/logs/contextsearch.log; other profiles log to the console only."
        ),
    )

    # ------------------------------------------------------------------ server
    host: str = field(
        default="127.0.0.1",
        metadata=_doc("Interface the server binds to. Use 0.0.0.0 to accept connections from other machines."),
    )
    port: int = field(default=8080, metadata=_doc("Port the server listens on."))
    api_key: str = field(
        default="",
        metadata=_doc(
            "When set, every /api request must send it as X-API-Key or Authorization: Bearer. Leave empty to run without auth.",
            secret=True,
        ),
    )
    cors_origins: str = field(
        default="",
        metadata=_doc(
            "Comma-separated origins allowed to call the API from a browser on another host. Empty disables CORS headers."
        ),
    )
    allowed_roots: str = field(
        default="",
        metadata=_doc(
            "Comma-separated folders that may be added as sources. Every source path has to sit inside one of them. Empty allows any path in development and refuses all folder sources in production."
        ),
    )
    max_upload_mb: int = field(default=50, metadata=_doc("Largest file accepted by the upload endpoint, in MB."))

    # ------------------------------------------------------------------ ingest
    include_extensions: str = field(
        default="pdf,docx,pptx,txt,md,markdown,rst,html,htm,csv,log",
        metadata=_doc(
            "File extensions that get indexed from watched folders and uploads. Image types are added automatically when OCR is enabled."
        ),
    )
    exclude_globs: str = field(
        default="**/.git/**,**/node_modules/**,**/__pycache__/**,**/~$*,**/.~lock.*,**/.DS_Store",
        metadata=_doc(
            "Comma-separated glob patterns that are skipped while scanning. Office lock files and VCS folders are skipped by default."
        ),
    )
    max_file_mb: int = field(default=200, metadata=_doc("Files larger than this are skipped and reported, not read."))
    scan_interval_seconds: int = field(
        default=300,
        metadata=_doc(
            "How often every enabled folder source is rescanned for changes. 0 turns periodic scans off; file system events still trigger indexing."
        ),
    )
    watch_folders: bool = field(
        default=True,
        metadata=_doc(
            "Subscribe to file system events (inotify on Linux, ReadDirectoryChangesW on Windows) so changes are picked up within seconds."
        ),
    )
    watch_debounce_seconds: float = field(
        default=2.0,
        metadata=_doc(
            "Quiet period after the last event on a file before it is indexed. Editors and copies write in bursts."
        ),
    )
    ocr: bool = field(
        default=False, metadata=_doc("Run OCR on images and on PDF pages that have no text layer. Needs the ocr extra.")
    )
    ocr_min_page_chars: int = field(
        default=20,
        metadata=_doc(
            "A PDF page with fewer extracted characters than this is treated as scanned and sent to OCR when OCR is on."
        ),
    )
    ocr_render_scale: float = field(
        default=2.0,
        metadata=_doc("Zoom factor used to render a PDF page to an image before OCR. 2.0 is about 144 dpi."),
    )

    # ------------------------------------------------------------------ chunking
    chunker: str = field(
        default="recursive",
        metadata=_doc(
            "Chunking strategy: recursive splits on paragraphs and sentences, fixed cuts every N words. A custom chunker registered in code is named here too.",
            chunking=True,
        ),
    )
    chunk_words: int = field(
        default=200,
        metadata=_doc(
            "Target chunk length in words. Around 200 words fits comfortably inside a 512-token embedding window.",
            chunking=True,
        ),
    )
    chunk_overlap_words: int = field(
        default=30,
        metadata=_doc(
            "Words carried over from the end of one chunk into the start of the next, so sentences at a boundary are searchable in both.",
            chunking=True,
        ),
    )
    chunk_min_words: int = field(
        default=40,
        metadata=_doc("Trailing pieces shorter than this are merged into the previous chunk.", chunking=True),
    )

    # ------------------------------------------------------------------ embedding
    embedding_models: str = field(
        default="BAAI/bge-small-en-v1.5",
        metadata=_doc(
            "Comma-separated sentence-transformers models to keep dense indexes for. Empty runs keyword search only. Each model costs one embedding pass per chunk."
        ),
    )
    embedding_device: str = field(
        default="",
        metadata=_doc(
            "Device for embedding and reranking models: cpu, cuda, cuda:1, mps. Empty picks cuda when available, otherwise cpu."
        ),
    )
    embedding_batch_size: int = field(
        default=32, metadata=_doc("Chunks encoded per forward pass. Raise it on a GPU, lower it if memory is tight.")
    )
    embedding_max_seq_length: int = field(
        default=0, metadata=_doc("Override the model's token limit. 0 keeps the model default.")
    )
    embedding_query_prefix: str = field(
        default="",
        metadata=_doc(
            "Text prepended to queries for models without a built-in preset. Known families (bge, e5, nomic) get the right prefix automatically."
        ),
    )
    embedding_document_prefix: str = field(
        default="", metadata=_doc("Text prepended to chunks for models without a built-in preset.")
    )
    embedding_cache_dir: str = field(
        default="",
        metadata=_doc(
            "Where model weights are downloaded to. Empty uses the Hugging Face default (~/.cache/huggingface)."
        ),
    )
    query_cache_size: int = field(
        default=512, metadata=_doc("Query embeddings kept in memory so repeated searches skip the model.")
    )

    # ------------------------------------------------------------------ search
    default_strategy: str = field(
        default="hybrid", metadata=_doc("Strategy used when a request does not name one: lexical, dense or hybrid.")
    )
    default_top_k: int = field(default=10, metadata=_doc("Results returned when a request does not say."))
    max_top_k: int = field(default=100, metadata=_doc("Upper bound a client may ask for."))
    candidates: int = field(
        default=50,
        metadata=_doc(
            "Candidates each retriever contributes before fusion. Higher gives better recall at the cost of a little latency."
        ),
    )
    fusion: str = field(
        default="rrf",
        metadata=_doc(
            "How hybrid combines retrievers: rrf (reciprocal rank fusion, needs no tuning) or weighted (min-max normalised scores blended with hybrid_alpha)."
        ),
    )
    rrf_k: int = field(
        default=60,
        metadata=_doc(
            "The k constant in 1/(k + rank). 60 is the value from the original paper and works well in practice."
        ),
    )
    hybrid_alpha: float = field(
        default=0.5,
        metadata=_doc("Weight of the dense score in weighted fusion, between 0 (keyword only) and 1 (semantic only)."),
    )
    rerank: bool = field(
        default=False,
        metadata=_doc(
            "Rerank fused candidates with a cross-encoder by default. Clients can turn it on per request too."
        ),
    )
    rerank_model: str = field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        metadata=_doc("Cross-encoder used for reranking. Needs the dense extra."),
    )
    rerank_candidates: int = field(default=30, metadata=_doc("How many fused candidates are scored by the reranker."))
    group_by_document: bool = field(
        default=True,
        metadata=_doc("Collapse chunks from the same document into one result that lists the other matching passages."),
    )
    snippet_chars: int = field(default=320, metadata=_doc("Length of the highlighted excerpt shown for each result."))
    lexical_stopwords: bool = field(
        default=True,
        metadata=_doc(
            "Drop common English words from keyword queries. They stay in the index, so quoted phrases still match them."
        ),
    )
    confidence_dense_low: float = field(
        default=0.35,
        metadata=_doc(
            "Cosine similarity that maps to 0 confidence for a semantic-only hit. Depends on the model; see docs/SEARCH.md for calibration."
        ),
    )
    confidence_dense_high: float = field(
        default=0.80, metadata=_doc("Cosine similarity that maps to full confidence for a semantic-only hit.")
    )
    confidence_strong: float = field(default=0.70, metadata=_doc("Confidence at or above this is labelled strong."))
    confidence_likely: float = field(
        default=0.40, metadata=_doc("Confidence at or above this is labelled likely; below is weak.")
    )
    query_log: bool = field(
        default=True,
        metadata=_doc(
            "Record every search (query text, strategy, latency, result count) in the database for the activity view."
        ),
    )
    query_log_keep: int = field(
        default=10000, metadata=_doc("How many recent searches to keep. Older rows are trimmed.")
    )

    # ------------------------------------------------------------------ vectors
    vector_compact_ratio: float = field(
        default=0.2,
        metadata=_doc("The in-memory matrix is rebuilt once this fraction of its rows belong to deleted chunks."),
    )

    # ------------------------------------------------------------------ derived
    @property
    def data_path(self) -> Path:
        return Path(self.data_dir).expanduser().resolve()

    @property
    def db_path(self) -> Path:
        return self.data_path / "contextsearch.db"

    @property
    def uploads_path(self) -> Path:
        return self.data_path / "uploads"

    @property
    def logs_path(self) -> Path:
        return self.data_path / "logs"

    @property
    def log_file_path(self) -> Path | None:
        if self.log_file == "auto":
            return self.logs_path / "contextsearch.log"
        if self.log_file:
            return Path(self.log_file).expanduser()
        return None

    @property
    def embedding_model_list(self) -> list[str]:
        return _split_csv(self.embedding_models)

    @property
    def include_extension_set(self) -> set[str]:
        """Extensions that get indexed. Image types join the set when OCR is on."""
        exts = {e.lower().lstrip(".") for e in _split_csv(self.include_extensions)}
        if self.ocr:
            exts |= IMAGE_EXTENSIONS
        return exts

    @property
    def exclude_glob_list(self) -> list[str]:
        return _split_csv(self.exclude_globs)

    @property
    def allowed_root_list(self) -> list[Path]:
        return [Path(p).expanduser().resolve() for p in _split_csv(self.allowed_roots)]

    @property
    def cors_origin_list(self) -> list[str]:
        return _split_csv(self.cors_origins)

    @property
    def is_production(self) -> bool:
        return self.profile == "production"

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_env(cls, env: dict[str, str] | None = None, dotenv: bool = True, **overrides: Any) -> Settings:
        """Build settings from defaults, profile, environment and explicit overrides."""
        env = dict(os.environ if env is None else env)
        if dotenv:
            for key, value in _read_dotenv(Path(".env")).items():
                env.setdefault(key, value)

        profile = overrides.get("profile") or env.get(ENV_PREFIX + "PROFILE") or "development"
        values: dict[str, Any] = dict(_profile_defaults(profile))
        values["profile"] = profile

        for f in fields(cls):
            if not f.metadata.get("env", True):
                continue
            raw = env.get(ENV_PREFIX + f.name.upper())
            if raw is not None:
                values[f.name] = _coerce(f.name, f.type, raw)
        values.update({k: v for k, v in overrides.items() if v is not None})
        settings = cls(**values)
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.profile not in PROFILES:
            raise ConfigurationError(f"profile must be one of {PROFILES}, got {self.profile!r}")
        if self.default_strategy not in STRATEGIES:
            raise ConfigurationError(f"default_strategy must be one of {STRATEGIES}, got {self.default_strategy!r}")
        if self.fusion not in FUSIONS:
            raise ConfigurationError(f"fusion must be one of {FUSIONS}, got {self.fusion!r}")
        if self.log_format not in ("text", "json"):
            raise ConfigurationError("log_format must be text or json")
        if not 1 <= self.port <= 65535:
            raise ConfigurationError("port must be between 1 and 65535")
        if self.chunk_words < 20:
            raise ConfigurationError("chunk_words must be at least 20")
        if not 0 <= self.chunk_overlap_words < self.chunk_words:
            raise ConfigurationError("chunk_overlap_words must be smaller than chunk_words")
        if self.chunk_min_words > self.chunk_words:
            raise ConfigurationError("chunk_min_words cannot exceed chunk_words")
        if not 0.0 <= self.hybrid_alpha <= 1.0:
            raise ConfigurationError("hybrid_alpha must be between 0 and 1")
        if self.rrf_k < 1:
            raise ConfigurationError("rrf_k must be at least 1")
        if self.default_top_k < 1 or self.max_top_k < self.default_top_k:
            raise ConfigurationError("default_top_k must be at least 1 and no larger than max_top_k")
        if self.candidates < 1:
            raise ConfigurationError("candidates must be at least 1")
        if self.confidence_dense_high <= self.confidence_dense_low:
            raise ConfigurationError("confidence_dense_high must be greater than confidence_dense_low")
        if self.confidence_likely > self.confidence_strong:
            raise ConfigurationError("confidence_likely cannot exceed confidence_strong")
        if self.embedding_batch_size < 1:
            raise ConfigurationError("embedding_batch_size must be at least 1")

    # ------------------------------------------------------------------ helpers
    def chunking_signature(self) -> str:
        """Stable hash of every parameter that changes how text is chunked."""
        parts = {f.name: getattr(self, f.name) for f in fields(self) if f.metadata.get("chunking")}
        parts["algorithm_version"] = CHUNKING_ALGORITHM_VERSION
        return hashlib.sha1(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:12]

    def to_dict(self, include_secrets: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if f.metadata.get("secret") and not include_secrets:
                value = "***" if value else ""
            out[f.name] = value
        return out

    def ensure_dirs(self) -> None:
        for path in (self.data_path, self.uploads_path, self.logs_path):
            path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def describe(cls) -> list[dict[str, Any]]:
        """Rows for the generated configuration reference."""
        rows = []
        for f in fields(cls):
            if f.default is MISSING:
                continue
            rows.append(
                {
                    "name": f.name,
                    "env": ENV_PREFIX + f.name.upper(),
                    "type": _type_name(f.type),
                    "default": "" if f.metadata.get("secret") else f.default,
                    "help": f.metadata.get("help", ""),
                    "chunking": bool(f.metadata.get("chunking")),
                }
            )
        return rows

    @classmethod
    def markdown_reference(cls) -> str:
        """The body of docs/CONFIGURATION.md."""
        lines = [
            "# Configuration",
            "",
            "Every setting is read from an environment variable with the `CSE_` prefix, or from a `.env` file in the folder you start the server from.",
            "Profiles set sensible groups of defaults; anything you set explicitly wins over the profile.",
            "",
            "This file is generated from `contextsearch/config.py` with `contextsearch config --markdown > docs/CONFIGURATION.md`. Edit the field help there, not here.",
            "",
            "## Profiles",
            "",
            "| Profile | What changes |",
            "|---|---|",
            "| `development` (default) | text logs on the console, binds to `127.0.0.1`, any folder can be added as a source |",
            "| `production` | json logs written to `<data_dir>/logs/contextsearch.log`, binds to `0.0.0.0`, folder sources must sit inside `CSE_ALLOWED_ROOTS` |",
            "| `testing` | what the test suite uses: no watcher, no periodic scans, no query log, no embedding models |",
            "",
            "Select one with `CSE_PROFILE=production` or `contextsearch serve --profile production`.",
            "",
            "## Settings",
            "",
            "| Setting | Environment variable | Type | Default | What it does |",
            "|---|---|---|---|---|",
        ]
        for row in cls.describe():
            default = row["default"]
            default_text = "" if default in ("", None) else f"`{default}`"
            help_text = row["help"]
            if row["chunking"]:
                help_text += " Part of the chunking signature: changing it means existing chunks were cut differently, and `contextsearch reindex` rebuilds them."
            lines.append(f"| `{row['name']}` | `{row['env']}` | {row['type']} | {default_text} | {help_text} |")
        lines += [
            "",
            "## Derived paths",
            "",
            "| Path | Location |",
            "|---|---|",
            "| database | `<data_dir>/contextsearch.db` (plus `-wal` and `-shm` files while the server runs) |",
            "| uploads | `<data_dir>/uploads/<YYYYMMDD>/` |",
            "| logs | `<data_dir>/logs/contextsearch.log` in production, or wherever `log_file` points |",
            "",
            "## Examples",
            "",
            "```bash",
            "# keyword search only, no model download at all",
            "CSE_EMBEDDING_MODELS= contextsearch serve",
            "",
            "# two dense indexes to compare, reranking on, served on the LAN",
            "CSE_EMBEDDING_MODELS=BAAI/bge-small-en-v1.5,sentence-transformers/all-MiniLM-L6-v2 \\",
            "CSE_RERANK=true CSE_HOST=0.0.0.0 CSE_API_KEY=change-me contextsearch serve",
            "",
            "# production: locked-down roots, json logs",
            "CSE_PROFILE=production CSE_ALLOWED_ROOTS=/srv/docs,/mnt/share CSE_API_KEY=change-me contextsearch serve",
            "```",
            "",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------- helpers


def _profile_defaults(profile: str) -> dict[str, Any]:
    if profile == "production":
        return {"log_format": "json", "host": "0.0.0.0", "log_file": "auto"}
    if profile == "testing":
        return {"watch_folders": False, "scan_interval_seconds": 0, "query_log": False, "embedding_models": ""}
    return {}


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in (value or "").split(",") if part.strip()]


def _type_name(annotation: Any) -> str:
    name = annotation if isinstance(annotation, str) else getattr(annotation, "__name__", str(annotation))
    return {"str": "string", "int": "integer", "float": "number", "bool": "boolean"}.get(name, name)


def _coerce(name: str, annotation: Any, raw: str) -> Any:
    kind = annotation if isinstance(annotation, str) else getattr(annotation, "__name__", "str")
    raw = raw.strip()
    try:
        if kind == "bool":
            if raw.lower() in ("1", "true", "yes", "on"):
                return True
            if raw.lower() in ("0", "false", "no", "off", ""):
                return False
            raise ValueError(raw)
        if kind == "int":
            return int(raw)
        if kind == "float":
            return float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{ENV_PREFIX}{name.upper()}: cannot read {raw!r} as {kind}") from exc
    return raw


def _read_dotenv(path: Path) -> dict[str, str]:
    """A small .env reader: KEY=VALUE lines, # comments, optional quotes."""
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[7:].strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key.startswith(ENV_PREFIX):
            values[key] = value
    return values
