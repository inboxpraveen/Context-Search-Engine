"""SQLite storage: one file is the source of truth for everything indexed.

Tables:

* ``sources``: watched folders and the uploads area.
* ``documents``: one row per file, with size, mtime, content hash, revision
  and status. Removed files keep their row (status ``removed``) so the
  history stays readable, but their pages, chunks and vectors are dropped.
* ``revisions``: one row per indexed revision of a document.
* ``pages``: the normalised page text the viewer shows and chunks point into.
* ``chunks``: verbatim slices of pages with character offsets. Ids are never
  reused (AUTOINCREMENT), so a chunk id always means one specific piece of
  text, also in the in-memory vector indexes.
* ``chunks_fts``: FTS5 index over chunk text (external content, kept in sync
  by triggers) that gives BM25 ranking, highlighting and snippets.
* ``models`` and ``embeddings``: one vector per (chunk, model), so several
  embedding models can be indexed and compared side by side.
* ``events``: what happened to which file and when.
* ``queries``: recent searches with their latency, for the activity view.

The database runs in WAL mode so searches keep going while a folder is being
indexed. Writes go through one lock; SQLite only allows one writer anyway and
the lock turns "database is locked" errors into a short wait.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .errors import ConfigurationError, ConflictError, NotFoundError, ValidationError
from .observability import get_logger

log = get_logger("contextsearch.store")

SCHEMA_VERSION = 1
HL_OPEN = ""  # private-use characters mark FTS highlights; they cannot occur in text
HL_CLOSE = ""

DOCUMENT_STATUSES = ("indexed", "failed", "removed", "pending")
DOCUMENT_SORTS = {
    "indexed_at": "d.indexed_at",
    "updated_at": "d.updated_at",
    "filename": "d.filename COLLATE NOCASE",
    "size": "d.size",
    "pages": "d.pages",
    "chunks": "d.chunks",
    "revision": "d.revision",
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS sources (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    path TEXT NOT NULL UNIQUE,
    recursive INTEGER NOT NULL DEFAULT 1,
    include TEXT NOT NULL DEFAULT '',
    exclude TEXT NOT NULL DEFAULT '',
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL,
    last_scan_at REAL,
    last_scan TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    path TEXT NOT NULL UNIQUE,
    relpath TEXT NOT NULL,
    filename TEXT NOT NULL,
    ext TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    size INTEGER NOT NULL DEFAULT 0,
    mtime REAL NOT NULL DEFAULT 0,
    content_hash TEXT NOT NULL DEFAULT '',
    revision INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    error TEXT NOT NULL DEFAULT '',
    reader TEXT NOT NULL DEFAULT '',
    pages INTEGER NOT NULL DEFAULT 0,
    chunks INTEGER NOT NULL DEFAULT 0,
    chars INTEGER NOT NULL DEFAULT 0,
    ocr_pages INTEGER NOT NULL DEFAULT 0,
    chunking_signature TEXT NOT NULL DEFAULT '',
    first_seen_at REAL NOT NULL,
    indexed_at REAL,
    updated_at REAL NOT NULL,
    removed_at REAL,
    metadata TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS documents_source ON documents(source_id, status);
CREATE INDEX IF NOT EXISTS documents_status ON documents(status, updated_at);
CREATE INDEX IF NOT EXISTS documents_ext ON documents(ext);

CREATE TABLE IF NOT EXISTS revisions (
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    revision INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    size INTEGER NOT NULL,
    mtime REAL NOT NULL,
    pages INTEGER NOT NULL,
    chunks INTEGER NOT NULL,
    indexed_at REAL NOT NULL,
    PRIMARY KEY (document_id, revision)
);

CREATE TABLE IF NOT EXISTS pages (
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page INTEGER NOT NULL,
    text TEXT NOT NULL,
    ocr INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (document_id, page)
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    char_start INTEGER NOT NULL,
    char_end INTEGER NOT NULL,
    word_count INTEGER NOT NULL,
    text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_document ON chunks(document_id, chunk_index);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, content='chunks', content_rowid='id', tokenize='porter unicode61'
);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TABLE IF NOT EXISTS models (
    name TEXT PRIMARY KEY,
    dimension INTEGER NOT NULL,
    kind TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    model TEXT NOT NULL REFERENCES models(name) ON DELETE CASCADE,
    vector BLOB NOT NULL,
    PRIMARY KEY (chunk_id, model)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS embeddings_model ON embeddings(model, chunk_id);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    ts REAL NOT NULL,
    kind TEXT NOT NULL,
    source_id TEXT,
    document_id TEXT,
    path TEXT,
    detail TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS events_ts ON events(ts);

CREATE TABLE IF NOT EXISTS queries (
    id INTEGER PRIMARY KEY,
    ts REAL NOT NULL,
    request_id TEXT NOT NULL,
    query TEXT NOT NULL,
    strategy TEXT NOT NULL,
    models TEXT NOT NULL DEFAULT '',
    rerank INTEGER NOT NULL DEFAULT 0,
    top_k INTEGER NOT NULL,
    results INTEGER NOT NULL,
    latency_ms REAL NOT NULL,
    timings TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS queries_ts ON queries(ts);
"""


def new_id() -> str:
    return uuid.uuid4().hex[:16]


# ------------------------------------------------------------------ records


@dataclass
class Source:
    id: str
    kind: str
    name: str
    path: str
    recursive: bool = True
    include: str = ""
    exclude: str = ""
    enabled: bool = True
    created_at: float = 0.0
    last_scan_at: float | None = None
    last_scan: dict[str, Any] = field(default_factory=dict)
    document_count: int = 0
    chunk_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "name": self.name,
            "path": self.path,
            "recursive": self.recursive,
            "include": self.include,
            "exclude": self.exclude,
            "enabled": self.enabled,
            "created_at": self.created_at,
            "last_scan_at": self.last_scan_at,
            "last_scan": self.last_scan,
            "document_count": self.document_count,
            "chunk_count": self.chunk_count,
        }


@dataclass
class Document:
    id: str
    source_id: str
    path: str
    relpath: str
    filename: str
    ext: str
    title: str = ""
    size: int = 0
    mtime: float = 0.0
    content_hash: str = ""
    revision: int = 0
    status: str = "pending"
    error: str = ""
    reader: str = ""
    pages: int = 0
    chunks: int = 0
    chars: int = 0
    ocr_pages: int = 0
    chunking_signature: str = ""
    first_seen_at: float = 0.0
    indexed_at: float | None = None
    updated_at: float = 0.0
    removed_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_path: bool = True) -> dict[str, Any]:
        data = {
            "id": self.id,
            "source_id": self.source_id,
            "relpath": self.relpath,
            "filename": self.filename,
            "ext": self.ext,
            "title": self.title,
            "size": self.size,
            "mtime": self.mtime,
            "content_hash": self.content_hash,
            "revision": self.revision,
            "status": self.status,
            "error": self.error,
            "reader": self.reader,
            "pages": self.pages,
            "chunks": self.chunks,
            "chars": self.chars,
            "ocr_pages": self.ocr_pages,
            "first_seen_at": self.first_seen_at,
            "indexed_at": self.indexed_at,
            "updated_at": self.updated_at,
            "removed_at": self.removed_at,
            "metadata": self.metadata,
        }
        if include_path:
            data["path"] = self.path
        return data


@dataclass
class ChunkRecord:
    id: int
    document_id: str
    page: int
    chunk_index: int
    char_start: int
    char_end: int
    word_count: int
    text: str


@dataclass
class ScanSnapshot:
    """What the sync loop needs to know about a stored document to diff it."""

    id: str
    size: int
    mtime: float
    content_hash: str
    status: str
    chunking_signature: str


# ------------------------------------------------------------------ store


class Store:
    def __init__(self, path: Path | str, readonly: bool = False):
        self.path = Path(path)
        self.readonly = readonly
        self._local = threading.local()
        self._write_lock = threading.RLock()
        self._connections: list[sqlite3.Connection] = []
        self._connections_lock = threading.Lock()
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._memory_conn: sqlite3.Connection | None = None
        self._init_schema()

    # ---------------------------------------------------------------- connections
    def _connect(self) -> sqlite3.Connection:
        if str(self.path) == ":memory:":
            # One shared connection; tests only. check_same_thread is off because
            # the ingest worker thread may touch it.
            if self._memory_conn is None:
                self._memory_conn = sqlite3.connect(":memory:", check_same_thread=False, isolation_level=None)
                self._memory_conn.row_factory = sqlite3.Row
                self._memory_conn.execute("PRAGMA foreign_keys=ON")
            return self._memory_conn
        conn = sqlite3.connect(str(self.path), timeout=30, isolation_level=None, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA cache_size=-65536")
        conn.execute("PRAGMA busy_timeout=15000")
        try:
            conn.execute("PRAGMA mmap_size=268435456")
        except sqlite3.DatabaseError:
            pass
        with self._connections_lock:
            self._connections.append(conn)
        return conn

    @property
    def conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._connect()
            self._local.conn = conn
        return conn

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Serialised write transaction. Nested calls join the outer transaction."""
        with self._write_lock:
            conn = self.conn
            if conn.in_transaction:
                yield conn
                return
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            else:
                conn.execute("COMMIT")

    def close(self) -> None:
        with self._connections_lock:
            conns = list(self._connections)
            self._connections.clear()
        for conn in conns:
            try:
                conn.close()
            except sqlite3.Error:
                pass
        if self._memory_conn is not None:
            self._memory_conn.close()
            self._memory_conn = None
        self._local = threading.local()

    # ---------------------------------------------------------------- schema
    def _init_schema(self) -> None:
        conn = self.conn
        try:
            has_meta = conn.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name='meta'").fetchone()[
                0
            ]
        except sqlite3.DatabaseError as exc:
            raise ConfigurationError(f"{self.path} is not a usable SQLite database: {exc}") from exc
        if self.readonly:
            return
        # executescript runs outside our transaction helper: it commits first and
        # every statement is idempotent (IF NOT EXISTS), so that is fine.
        with self._write_lock:
            try:
                conn.executescript(_SCHEMA)
            except sqlite3.OperationalError as exc:
                if "fts5" in str(exc).lower():
                    raise ConfigurationError(
                        "this Python build has SQLite without FTS5; install Python from python.org or conda"
                    ) from exc
                raise
        if has_meta:
            version = int(self.get_meta("schema_version") or 0)
            if version > SCHEMA_VERSION:
                raise ConfigurationError(
                    f"database schema {version} is newer than this version understands ({SCHEMA_VERSION})"
                )
        self.set_meta("schema_version", str(SCHEMA_VERSION))
        if not self.get_meta("created_at"):
            self.set_meta("created_at", str(time.time()))

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO meta(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )

    # ---------------------------------------------------------------- sources
    def add_source(
        self, kind: str, name: str, path: str, recursive: bool = True, include: str = "", exclude: str = ""
    ) -> Source:
        if kind not in ("folder", "uploads"):
            raise ValidationError(f"unknown source kind {kind!r}")
        existing = self.conn.execute("SELECT id FROM sources WHERE path=?", (path,)).fetchone()
        if existing:
            raise ConflictError("this folder is already a source", {"source_id": existing[0]})
        source = Source(
            id=new_id(),
            kind=kind,
            name=name,
            path=path,
            recursive=recursive,
            include=include,
            exclude=exclude,
            created_at=time.time(),
        )
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO sources(id, kind, name, path, recursive, include, exclude, enabled, created_at) VALUES (?,?,?,?,?,?,?,1,?)",
                (source.id, kind, name, path, int(recursive), include, exclude, source.created_at),
            )
            self.add_event("source_added", source_id=source.id, path=path, detail={"kind": kind, "name": name})
        return source

    def _source_from_row(self, row: sqlite3.Row) -> Source:
        return Source(
            id=row["id"],
            kind=row["kind"],
            name=row["name"],
            path=row["path"],
            recursive=bool(row["recursive"]),
            include=row["include"],
            exclude=row["exclude"],
            enabled=bool(row["enabled"]),
            created_at=row["created_at"],
            last_scan_at=row["last_scan_at"],
            last_scan=json.loads(row["last_scan"] or "{}"),
            document_count=row["document_count"],
            chunk_count=row["chunk_count"],
        )

    _SOURCE_SELECT = """
        SELECT s.*,
               (SELECT count(*) FROM documents d WHERE d.source_id = s.id AND d.status = 'indexed') AS document_count,
               (SELECT coalesce(sum(d.chunks), 0) FROM documents d WHERE d.source_id = s.id AND d.status = 'indexed') AS chunk_count
        FROM sources s
    """

    def list_sources(self) -> list[Source]:
        rows = self.conn.execute(self._SOURCE_SELECT + " ORDER BY s.created_at").fetchall()
        return [self._source_from_row(r) for r in rows]

    def get_source(self, source_id: str) -> Source:
        row = self.conn.execute(self._SOURCE_SELECT + " WHERE s.id=?", (source_id,)).fetchone()
        if not row:
            raise NotFoundError(f"source {source_id} not found")
        return self._source_from_row(row)

    def get_source_by_path(self, path: str) -> Source | None:
        row = self.conn.execute(self._SOURCE_SELECT + " WHERE s.path=?", (path,)).fetchone()
        return self._source_from_row(row) if row else None

    def remove_source(self, source_id: str) -> Source:
        source = self.get_source(source_id)
        with self.transaction() as conn:
            conn.execute("DELETE FROM sources WHERE id=?", (source_id,))
            self.add_event(
                "source_removed", source_id=source_id, path=source.path, detail={"documents": source.document_count}
            )
        return source

    def set_source_enabled(self, source_id: str, enabled: bool) -> Source:
        with self.transaction() as conn:
            conn.execute("UPDATE sources SET enabled=? WHERE id=?", (int(enabled), source_id))
        return self.get_source(source_id)

    def record_scan(self, source_id: str, summary: dict[str, Any]) -> None:
        with self.transaction() as conn:
            conn.execute(
                "UPDATE sources SET last_scan_at=?, last_scan=? WHERE id=?",
                (time.time(), json.dumps(summary), source_id),
            )

    # ---------------------------------------------------------------- documents
    def _document_from_row(self, row: sqlite3.Row) -> Document:
        return Document(
            id=row["id"],
            source_id=row["source_id"],
            path=row["path"],
            relpath=row["relpath"],
            filename=row["filename"],
            ext=row["ext"],
            title=row["title"],
            size=row["size"],
            mtime=row["mtime"],
            content_hash=row["content_hash"],
            revision=row["revision"],
            status=row["status"],
            error=row["error"],
            reader=row["reader"],
            pages=row["pages"],
            chunks=row["chunks"],
            chars=row["chars"],
            ocr_pages=row["ocr_pages"],
            chunking_signature=row["chunking_signature"],
            first_seen_at=row["first_seen_at"],
            indexed_at=row["indexed_at"],
            updated_at=row["updated_at"],
            removed_at=row["removed_at"],
            metadata=json.loads(row["metadata"] or "{}"),
        )

    def get_document(self, document_id: str) -> Document:
        row = self.conn.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone()
        if not row:
            raise NotFoundError(f"document {document_id} not found")
        return self._document_from_row(row)

    def get_document_by_path(self, path: str) -> Document | None:
        row = self.conn.execute("SELECT * FROM documents WHERE path=?", (path,)).fetchone()
        return self._document_from_row(row) if row else None

    def get_documents(self, ids: list[str]) -> dict[str, Document]:
        if not ids:
            return {}
        out: dict[str, Document] = {}
        for i in range(0, len(ids), 500):
            batch = ids[i : i + 500]
            marks = ",".join("?" * len(batch))
            for row in self.conn.execute(f"SELECT * FROM documents WHERE id IN ({marks})", batch):
                out[row["id"]] = self._document_from_row(row)
        return out

    def list_documents(
        self,
        source_id: str | None = None,
        status: str | None = "indexed",
        q: str | None = None,
        ext: str | None = None,
        sort: str = "indexed_at",
        descending: bool = True,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Document], int]:
        where, params = self._document_filters(source_id, status, ext)
        if q:
            where.append("(d.filename LIKE ? OR d.title LIKE ? OR d.relpath LIKE ?)")
            like = f"%{q}%"
            params += [like, like, like]
        clause = (" WHERE " + " AND ".join(where)) if where else ""
        order = DOCUMENT_SORTS.get(sort, "d.indexed_at")
        direction = "DESC" if descending else "ASC"
        total = self.conn.execute(f"SELECT count(*) FROM documents d{clause}", params).fetchone()[0]
        rows = self.conn.execute(
            f"SELECT d.* FROM documents d{clause} ORDER BY {order} {direction}, d.id LIMIT ? OFFSET ?",
            [*params, limit, offset],
        ).fetchall()
        return [self._document_from_row(r) for r in rows], total

    @staticmethod
    def _document_filters(source_id: str | None, status: str | None, ext: str | None) -> tuple[list[str], list[Any]]:
        where: list[str] = []
        params: list[Any] = []
        if source_id:
            where.append("d.source_id = ?")
            params.append(source_id)
        if status and status != "all":
            where.append("d.status = ?")
            params.append(status)
        if ext:
            where.append("d.ext = ?")
            params.append(ext.lower().lstrip("."))
        return where, params

    def snapshot_for_source(self, source_id: str) -> dict[str, ScanSnapshot]:
        rows = self.conn.execute(
            "SELECT id, path, size, mtime, content_hash, status, chunking_signature FROM documents WHERE source_id=? AND status != 'removed'",
            (source_id,),
        )
        return {
            r["path"]: ScanSnapshot(
                r["id"], r["size"], r["mtime"], r["content_hash"], r["status"], r["chunking_signature"]
            )
            for r in rows
        }

    def documents_with_signature_other_than(self, signature: str, limit: int = 100000) -> list[Document]:
        rows = self.conn.execute(
            "SELECT * FROM documents WHERE status='indexed' AND chunking_signature != ? LIMIT ?", (signature, limit)
        ).fetchall()
        return [self._document_from_row(r) for r in rows]

    def upsert_pending(self, source: Source, path: str, relpath: str, size: int, mtime: float) -> Document:
        """Make sure a row exists for a file that is about to be read."""
        now = time.time()
        existing = self.get_document_by_path(path)
        if existing:
            return existing
        filename = Path(path).name
        ext = Path(path).suffix.lower().lstrip(".")
        doc = Document(
            id=new_id(),
            source_id=source.id,
            path=path,
            relpath=relpath,
            filename=filename,
            ext=ext,
            size=size,
            mtime=mtime,
            status="pending",
            first_seen_at=now,
            updated_at=now,
        )
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO documents(id, source_id, path, relpath, filename, ext, size, mtime, status, first_seen_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (doc.id, doc.source_id, path, relpath, filename, ext, size, mtime, "pending", now, now),
            )
        return doc

    def store_indexed(
        self,
        document_id: str,
        *,
        title: str,
        size: int,
        mtime: float,
        content_hash: str,
        reader: str,
        pages: list[tuple[int, str, bool]],
        chunks: list[tuple[int, int, int, int, int, str]],
        chunking_signature: str,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Document, list[int]]:
        """Replace a document's content with a freshly parsed revision.

        ``pages`` rows are (page, text, ocr). ``chunks`` rows are
        (page, chunk_index, char_start, char_end, word_count, text). Returns
        the updated document and the new chunk ids in order.
        """
        now = time.time()
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT revision, status, content_hash FROM documents WHERE id=?", (document_id,)
            ).fetchone()
            if not row:
                raise NotFoundError(f"document {document_id} not found")
            revision = int(row["revision"]) + 1
            previous_status = row["status"]
            conn.execute("DELETE FROM chunks WHERE document_id=?", (document_id,))
            conn.execute("DELETE FROM pages WHERE document_id=?", (document_id,))
            conn.executemany(
                "INSERT INTO pages(document_id, page, text, ocr) VALUES (?,?,?,?)",
                [(document_id, page, text, int(ocr)) for page, text, ocr in pages],
            )
            chunk_ids: list[int] = []
            for page, index, start, end, words, text in chunks:
                cur = conn.execute(
                    "INSERT INTO chunks(document_id, page, chunk_index, char_start, char_end, word_count, text) VALUES (?,?,?,?,?,?,?)",
                    (document_id, page, index, start, end, words, text),
                )
                chunk_ids.append(int(cur.lastrowid))
            chars = sum(len(t) for _, t, _ in pages)
            ocr_pages = sum(1 for _, _, o in pages if o)
            conn.execute(
                """UPDATE documents SET title=?, size=?, mtime=?, content_hash=?, revision=?, status='indexed', error='', reader=?,
                   pages=?, chunks=?, chars=?, ocr_pages=?, chunking_signature=?, indexed_at=?, updated_at=?, removed_at=NULL, metadata=?
                   WHERE id=?""",
                (
                    title[:300],
                    size,
                    mtime,
                    content_hash,
                    revision,
                    reader,
                    len(pages),
                    len(chunks),
                    chars,
                    ocr_pages,
                    chunking_signature,
                    now,
                    now,
                    json.dumps(metadata or {}),
                    document_id,
                ),
            )
            conn.execute(
                "INSERT OR REPLACE INTO revisions(document_id, revision, content_hash, size, mtime, pages, chunks, indexed_at) VALUES (?,?,?,?,?,?,?,?)",
                (document_id, revision, content_hash, size, mtime, len(pages), len(chunks), now),
            )
            doc = self.get_document(document_id)
            kind = "added" if revision == 1 or previous_status == "removed" else "updated"
            self.add_event(
                kind,
                source_id=doc.source_id,
                document_id=document_id,
                path=doc.path,
                detail={"revision": revision, "pages": len(pages), "chunks": len(chunks)},
            )
        return doc, chunk_ids

    def touch_document(self, document_id: str, size: int, mtime: float) -> None:
        """The file was rewritten with identical content; remember the new stat so it is not hashed again."""
        with self.transaction() as conn:
            conn.execute(
                "UPDATE documents SET size=?, mtime=?, updated_at=? WHERE id=?", (size, mtime, time.time(), document_id)
            )

    def mark_failed(
        self, document_id: str, error: str, size: int | None = None, mtime: float | None = None
    ) -> Document:
        now = time.time()
        with self.transaction() as conn:
            if size is not None and mtime is not None:
                conn.execute(
                    "UPDATE documents SET status='failed', error=?, size=?, mtime=?, updated_at=? WHERE id=?",
                    (error[:1000], size, mtime, now, document_id),
                )
            else:
                conn.execute(
                    "UPDATE documents SET status='failed', error=?, updated_at=? WHERE id=?",
                    (error[:1000], now, document_id),
                )
            doc = self.get_document(document_id)
            self.add_event(
                "failed", source_id=doc.source_id, document_id=document_id, path=doc.path, detail={"error": error[:300]}
            )
        return doc

    def mark_removed(self, document_id: str) -> Document:
        now = time.time()
        with self.transaction() as conn:
            conn.execute("DELETE FROM chunks WHERE document_id=?", (document_id,))
            conn.execute("DELETE FROM pages WHERE document_id=?", (document_id,))
            conn.execute(
                "UPDATE documents SET status='removed', chunks=0, removed_at=?, updated_at=? WHERE id=?",
                (now, now, document_id),
            )
            doc = self.get_document(document_id)
            self.add_event(
                "removed",
                source_id=doc.source_id,
                document_id=document_id,
                path=doc.path,
                detail={"revision": doc.revision},
            )
        return doc

    def delete_document(self, document_id: str) -> Document:
        doc = self.get_document(document_id)
        with self.transaction() as conn:
            conn.execute("DELETE FROM documents WHERE id=?", (document_id,))
            self.add_event("deleted", source_id=doc.source_id, document_id=document_id, path=doc.path, detail={})
        return doc

    def revisions(self, document_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM revisions WHERE document_id=? ORDER BY revision DESC", (document_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ---------------------------------------------------------------- pages and chunks
    def get_page(self, document_id: str, page: int) -> dict[str, Any]:
        row = self.conn.execute(
            "SELECT page, text, ocr FROM pages WHERE document_id=? AND page=?", (document_id, page)
        ).fetchone()
        if not row:
            raise NotFoundError(f"page {page} of document {document_id} not found")
        return {"page": row["page"], "text": row["text"], "ocr": bool(row["ocr"])}

    def list_page_numbers(self, document_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT page, length(text) AS chars, ocr FROM pages WHERE document_id=? ORDER BY page", (document_id,)
        ).fetchall()
        return [{"page": r["page"], "chars": r["chars"], "ocr": bool(r["ocr"])} for r in rows]

    def _chunk_from_row(self, row: sqlite3.Row) -> ChunkRecord:
        return ChunkRecord(
            row["id"],
            row["document_id"],
            row["page"],
            row["chunk_index"],
            row["char_start"],
            row["char_end"],
            row["word_count"],
            row["text"],
        )

    def get_chunks(self, ids: list[int]) -> dict[int, ChunkRecord]:
        out: dict[int, ChunkRecord] = {}
        ids = [int(i) for i in ids]
        for i in range(0, len(ids), 500):
            batch = ids[i : i + 500]
            marks = ",".join("?" * len(batch))
            for row in self.conn.execute(f"SELECT * FROM chunks WHERE id IN ({marks})", batch):
                out[row["id"]] = self._chunk_from_row(row)
        return out

    def chunks_for_document(self, document_id: str, page: int | None = None) -> list[ChunkRecord]:
        if page is None:
            rows = self.conn.execute(
                "SELECT * FROM chunks WHERE document_id=? ORDER BY chunk_index", (document_id,)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM chunks WHERE document_id=? AND page=? ORDER BY chunk_index", (document_id, page)
            ).fetchall()
        return [self._chunk_from_row(r) for r in rows]

    def allowed_chunk_ids(
        self, source_id: str | None = None, ext: str | None = None, document_id: str | None = None
    ) -> np.ndarray | None:
        """Chunk ids matching a filter, or None when there is no filter. Used to mask dense search."""
        if not (source_id or ext or document_id):
            return None
        where, params = self._document_filters(source_id, "indexed", ext)
        if document_id:
            where.append("d.id = ?")
            params.append(document_id)
        rows = self.conn.execute(
            f"SELECT c.id FROM chunks c JOIN documents d ON d.id = c.document_id WHERE {' AND '.join(where)}", params
        ).fetchall()
        return np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))

    # ---------------------------------------------------------------- lexical search
    def search_fts(
        self,
        match: str,
        limit: int,
        source_id: str | None = None,
        ext: str | None = None,
        document_id: str | None = None,
    ) -> list[tuple[int, float, str]]:
        """BM25-ranked chunk ids for an FTS5 MATCH expression, with highlighted text.

        Returns (chunk_id, score, highlighted_text). Scores are positive; larger
        is better (FTS5 reports negatives internally).
        """
        where = ["chunks_fts MATCH ?", "d.status = 'indexed'"]
        params: list[Any] = [match]
        extra, extra_params = self._document_filters(source_id, None, ext)
        where += extra
        params += extra_params
        if document_id:
            where.append("d.id = ?")
            params.append(document_id)
        sql = f"""
            SELECT c.id AS id, bm25(chunks_fts) AS score, highlight(chunks_fts, 0, ?, ?) AS hl
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.rowid
            JOIN documents d ON d.id = c.document_id
            WHERE {" AND ".join(where)}
            ORDER BY score
            LIMIT ?
        """
        rows = self.conn.execute(sql, [HL_OPEN, HL_CLOSE, *params, limit]).fetchall()
        return [(int(r["id"]), float(-r["score"]), r["hl"]) for r in rows]

    # ---------------------------------------------------------------- models and vectors
    def register_model(self, name: str, dimension: int, kind: str = "") -> None:
        with self.transaction() as conn:
            row = conn.execute("SELECT dimension FROM models WHERE name=?", (name,)).fetchone()
            if row and int(row[0]) != dimension:
                raise ConflictError(
                    f"model {name} is stored with dimension {row[0]}, the loaded model has {dimension}; drop it first",
                    {"model": name},
                )
            if not row:
                conn.execute(
                    "INSERT INTO models(name, dimension, kind, created_at) VALUES (?,?,?,?)",
                    (name, dimension, kind, time.time()),
                )
                self.add_event("model_added", detail={"model": name, "dimension": dimension})

    def list_models(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT m.*, (SELECT count(*) FROM embeddings e WHERE e.model = m.name) AS vectors FROM models m ORDER BY m.created_at"
        ).fetchall()
        return [dict(r) for r in rows]

    def remove_model(self, name: str) -> None:
        with self.transaction() as conn:
            cur = conn.execute("DELETE FROM models WHERE name=?", (name,))
            if cur.rowcount == 0:
                raise NotFoundError(f"model {name} has no index")
            self.add_event("model_removed", detail={"model": name})

    def store_vectors(self, model: str, chunk_ids: list[int], vectors: np.ndarray) -> None:
        if len(chunk_ids) != len(vectors):
            raise ValidationError("chunk_ids and vectors differ in length")
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        with self.transaction() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO embeddings(chunk_id, model, vector) VALUES (?,?,?)",
                [(int(cid), model, vec.tobytes()) for cid, vec in zip(chunk_ids, vectors, strict=True)],
            )

    def iter_vectors(self, model: str, batch: int = 20000) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield (chunk_ids, matrix) batches for a model, in chunk id order."""
        row = self.conn.execute("SELECT dimension FROM models WHERE name=?", (model,)).fetchone()
        if not row:
            return
        dim = int(row[0])
        cur = self.conn.execute("SELECT chunk_id, vector FROM embeddings WHERE model=? ORDER BY chunk_id", (model,))
        while True:
            rows = cur.fetchmany(batch)
            if not rows:
                break
            ids = np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))
            matrix = np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float32).reshape(len(rows), dim)
            yield ids, matrix

    def count_vectors(self, model: str) -> int:
        return int(self.conn.execute("SELECT count(*) FROM embeddings WHERE model=?", (model,)).fetchone()[0])

    def chunks_missing_vectors(self, model: str, limit: int = 1000) -> list[ChunkRecord]:
        rows = self.conn.execute(
            """SELECT c.* FROM chunks c
               JOIN documents d ON d.id = c.document_id AND d.status = 'indexed'
               WHERE NOT EXISTS (SELECT 1 FROM embeddings e WHERE e.chunk_id = c.id AND e.model = ?)
               ORDER BY c.id LIMIT ?""",
            (model, limit),
        ).fetchall()
        return [self._chunk_from_row(r) for r in rows]

    def count_chunks_missing_vectors(self, model: str) -> int:
        return int(
            self.conn.execute(
                """SELECT count(*) FROM chunks c JOIN documents d ON d.id = c.document_id AND d.status = 'indexed'
                   WHERE NOT EXISTS (SELECT 1 FROM embeddings e WHERE e.chunk_id = c.id AND e.model = ?)""",
                (model,),
            ).fetchone()[0]
        )

    # ---------------------------------------------------------------- events and queries
    def add_event(
        self,
        kind: str,
        source_id: str | None = None,
        document_id: str | None = None,
        path: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO events(ts, kind, source_id, document_id, path, detail) VALUES (?,?,?,?,?,?)",
                (time.time(), kind, source_id, document_id, path, json.dumps(detail or {})),
            )

    def list_events(
        self, limit: int = 100, kinds: list[str] | None = None, source_id: str | None = None
    ) -> list[dict[str, Any]]:
        where: list[str] = []
        params: list[Any] = []
        if kinds:
            where.append(f"kind IN ({','.join('?' * len(kinds))})")
            params += kinds
        if source_id:
            where.append("source_id = ?")
            params.append(source_id)
        clause = (" WHERE " + " AND ".join(where)) if where else ""
        rows = self.conn.execute(f"SELECT * FROM events{clause} ORDER BY id DESC LIMIT ?", [*params, limit]).fetchall()
        return [{**dict(r), "detail": json.loads(r["detail"] or "{}")} for r in rows]

    def trim_events(self, keep: int = 50000) -> None:
        with self.transaction() as conn:
            conn.execute("DELETE FROM events WHERE id <= (SELECT coalesce(max(id), 0) FROM events) - ?", (keep,))

    def log_query(
        self,
        request_id: str,
        query: str,
        strategy: str,
        models: list[str],
        rerank: bool,
        top_k: int,
        results: int,
        latency_ms: float,
        timings: dict[str, float],
    ) -> None:
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO queries(ts, request_id, query, strategy, models, rerank, top_k, results, latency_ms, timings) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    time.time(),
                    request_id,
                    query[:500],
                    strategy,
                    ",".join(models),
                    int(rerank),
                    top_k,
                    results,
                    latency_ms,
                    json.dumps(timings),
                ),
            )

    def recent_queries(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM queries ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [
            {
                **dict(r),
                "rerank": bool(r["rerank"]),
                "models": [m for m in r["models"].split(",") if m],
                "timings": json.loads(r["timings"] or "{}"),
            }
            for r in rows
        ]

    def trim_queries(self, keep: int) -> None:
        with self.transaction() as conn:
            conn.execute("DELETE FROM queries WHERE id <= (SELECT coalesce(max(id), 0) FROM queries) - ?", (keep,))

    def query_stats(self) -> dict[str, Any]:
        row = self.conn.execute(
            "SELECT count(*) AS n, avg(latency_ms) AS avg_ms, max(latency_ms) AS max_ms FROM queries"
        ).fetchone()
        return {
            "count": int(row["n"] or 0),
            "avg_ms": round(float(row["avg_ms"] or 0), 1),
            "max_ms": round(float(row["max_ms"] or 0), 1),
        }

    # ---------------------------------------------------------------- stats and maintenance
    def stats(self) -> dict[str, Any]:
        conn = self.conn
        counts = conn.execute(
            """SELECT
                 (SELECT count(*) FROM sources) AS sources,
                 (SELECT count(*) FROM documents WHERE status='indexed') AS documents,
                 (SELECT count(*) FROM documents WHERE status='failed') AS failed,
                 (SELECT count(*) FROM documents WHERE status='removed') AS removed,
                 (SELECT count(*) FROM chunks) AS chunks,
                 (SELECT coalesce(sum(pages), 0) FROM documents WHERE status='indexed') AS pages,
                 (SELECT coalesce(sum(chars), 0) FROM documents WHERE status='indexed') AS chars,
                 (SELECT count(*) FROM events) AS events"""
        ).fetchone()
        size_bytes = 0
        if str(self.path) != ":memory:" and self.path.exists():
            size_bytes = self.path.stat().st_size
            wal = self.path.with_name(self.path.name + "-wal")
            if wal.exists():
                size_bytes += wal.stat().st_size
        by_ext = {
            r["ext"]: r["n"]
            for r in conn.execute(
                "SELECT ext, count(*) AS n FROM documents WHERE status='indexed' GROUP BY ext ORDER BY n DESC"
            )
        }
        return {
            "sources": counts["sources"],
            "documents": counts["documents"],
            "failed_documents": counts["failed"],
            "removed_documents": counts["removed"],
            "pages": counts["pages"],
            "chunks": counts["chunks"],
            "chars": counts["chars"],
            "events": counts["events"],
            "database_bytes": size_bytes,
            "documents_by_type": by_ext,
            "models": self.list_models(),
            "queries": self.query_stats(),
        }

    def integrity_check(self) -> str:
        return str(self.conn.execute("PRAGMA integrity_check").fetchone()[0])

    def vacuum(self) -> None:
        with self._write_lock:
            self.conn.execute("VACUUM")

    def optimize_fts(self) -> None:
        with self.transaction() as conn:
            conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('optimize')")
