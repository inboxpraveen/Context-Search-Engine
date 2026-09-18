"""Shared fixtures: a testing-profile engine on a temp folder and small document makers."""

from __future__ import annotations

import io
import textwrap
from pathlib import Path

import pytest

from contextsearch import Engine, Settings


def make_settings(tmp_path: Path, **overrides) -> Settings:
    base = {
        "profile": "testing",
        "data_dir": str(tmp_path / "data"),
        "embedding_models": "test/hashing-64",
        "log_level": "WARNING",
    }
    base.update(overrides)
    return Settings.from_env(env={}, dotenv=False, **base)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path)


@pytest.fixture
def engine(settings: Settings):
    eng = Engine(settings, configure_logs=False)
    eng.start(warm=False)
    # Let the hashing model finish registering before tests add documents.
    for job in eng.jobs.list():
        job.wait(10)
    yield eng
    eng.close()


def write_text(folder: Path, relpath: str, text: str) -> Path:
    path = folder / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).strip() + "\n", encoding="utf-8")
    return path


def make_pdf(pages: list[str], title: str = "") -> bytes:
    """A minimal text-only PDF, enough for PDFium to extract page text from."""
    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    font = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    pages_obj = add(b"")
    page_ids = []
    for text in pages:
        lines = []
        for para in text.split("\n"):
            lines += textwrap.wrap(para, 90) or [""]
        escaped = [
            line.encode("latin-1", "replace").replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")
            for line in lines
        ]
        stream = b"BT /F1 11 Tf 50 760 Td 14 TL " + b" ".join(b"(" + line + b") Tj T*" for line in escaped) + b" ET"
        content = add(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
        page_ids.append(
            add(
                b"<< /Type /Page /Parent %d 0 R /MediaBox [0 0 612 792] /Contents %d 0 R /Resources << /Font << /F1 %d 0 R >> >> >>"
                % (pages_obj, content, font)
            )
        )
    objects[pages_obj - 1] = (
        b"<< /Type /Pages /Kids [" + b" ".join(b"%d 0 R" % i for i in page_ids) + b"] /Count %d >>" % len(page_ids)
    )
    info = add(b"<< /Title (" + title.encode("latin-1", "replace") + b") >>")
    catalog = add(b"<< /Type /Catalog /Pages %d 0 R >>" % pages_obj)
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(b"%d 0 obj\n" % number + body + b"\nendobj\n")
    xref = out.tell()
    out.write(b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1))
    for offset in offsets:
        out.write(b"%010d 00000 n \n" % offset)
    out.write(
        b"trailer\n<< /Size %d /Root %d 0 R /Info %d 0 R >>\nstartxref\n%d\n%%%%EOF\n"
        % (len(objects) + 1, catalog, info, xref)
    )
    return out.getvalue()


def wait_idle(engine: Engine, timeout: float = 30.0) -> None:
    for job in engine.jobs.list():
        job.wait(timeout)
