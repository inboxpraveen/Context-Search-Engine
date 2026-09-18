"""Documentation that is generated from or promised by the code must match it."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import pytest

from contextsearch.api import create_app
from contextsearch.config import Settings

ROOT = Path(__file__).resolve().parent.parent

# The Docker image ships the package and the tests but not the documentation.
pytestmark = pytest.mark.skipif(not (ROOT / "docs").is_dir(), reason="docs folder is not part of this checkout")


def test_configuration_reference_is_generated_from_settings():
    on_disk = (ROOT / "docs" / "CONFIGURATION.md").read_text(encoding="utf-8").replace("\r\n", "\n")
    assert on_disk == Settings.markdown_reference(), "run: contextsearch config --markdown > docs/CONFIGURATION.md"


def test_every_api_path_is_documented():
    settings = Settings.from_env(env={}, dotenv=False, profile="testing", data_dir=tempfile.mkdtemp())
    spec = create_app(settings=settings).openapi()
    api_md = (ROOT / "docs" / "API.md").read_text(encoding="utf-8")
    missing = [path for path in spec["paths"] if f"`{path}`" not in api_md]
    assert not missing, f"paths missing from docs/API.md: {missing}"


def test_no_dashes_in_prose_or_code():
    """The project writes with commas and full stops, not em or en dashes."""
    offenders = []
    for path in (
        list(ROOT.glob("*.md"))
        + list((ROOT / "docs").glob("*.md"))
        + list((ROOT / "examples").rglob("*"))
        + list((ROOT / "contextsearch").rglob("*"))
        + list((ROOT / "tests").glob("*.py"))
    ):
        if path.is_dir() or path.suffix in (".png", ".jpg", ".pyc"):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if re.search("[" + chr(0x2014) + chr(0x2013) + "]", text):
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, f"em or en dashes found in: {offenders}"
