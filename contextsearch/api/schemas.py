"""Request bodies. Response shapes are documented in docs/API.md and produced
by the engine's ``to_dict`` methods, which keeps one source of truth."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SearchBody(BaseModel):
    query: str = Field(
        ...,
        description="What to look for. Quote a phrase to require it exactly.",
        examples=["refund policy for late deliveries"],
    )
    strategy: str = Field("", description="lexical, dense or hybrid. Empty uses the server default.")
    models: list[str] | None = Field(
        None, description="Embedding models to use for dense retrieval. Empty uses every indexed model."
    )
    top_k: int = Field(0, ge=0, description="Results to return. 0 uses the server default.")
    candidates: int = Field(0, ge=0, description="Candidates per retriever before fusion. 0 uses the server default.")
    fusion: str = Field("", description="rrf or weighted.")
    rrf_k: int = Field(0, ge=0)
    alpha: float | None = Field(None, ge=0.0, le=1.0, description="Dense weight for weighted fusion.")
    rerank: bool | None = Field(None, description="Rerank fused candidates with a cross-encoder.")
    rerank_model: str = ""
    rerank_candidates: int = Field(0, ge=0)
    group_by_document: bool | None = None
    prefix_last: bool = Field(False, description="Treat the last word as a prefix, for search-as-you-type.")
    source_id: str | None = None
    ext: str | None = Field(None, description="Only documents with this extension, for example pdf.")
    document_id: str | None = Field(None, description="Search inside one document.")
    snippet_chars: int = Field(0, ge=0)
    min_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Drop results below this confidence.")
    include_text: bool = Field(True, description="Include the full chunk text in each result.")
    explain: bool = Field(True, description="Include evidence and the trace. Turn off for the smallest response.")

    def to_kwargs(self) -> dict[str, Any]:
        return self.model_dump()


class CompareVariant(BaseModel):
    label: str = ""
    strategy: str = ""
    models: list[str] | None = None
    fusion: str = ""
    alpha: float | None = None
    rerank: bool | None = None
    rerank_model: str = ""
    top_k: int = 0
    candidates: int = 0


class CompareBody(BaseModel):
    query: str
    variants: list[CompareVariant] = Field(..., min_length=1, max_length=4)
    source_id: str | None = None
    ext: str | None = None


class AddSourceBody(BaseModel):
    path: str = Field(
        ...,
        min_length=1,
        description="Folder to index and watch. Must sit inside an allowed root when those are configured.",
    )
    name: str = ""
    recursive: bool = True
    include: str = Field("", description="Comma-separated extensions that override the server default for this folder.")
    exclude: str = Field("", description="Extra comma-separated glob patterns to skip.")


class PatchSourceBody(BaseModel):
    enabled: bool


class BuildModelBody(BaseModel):
    name: str = Field(..., description="A sentence-transformers model id, for example BAAI/bge-small-en-v1.5.")


class ReindexBody(BaseModel):
    force: bool = Field(False, description="Reindex every document, not only those chunked with older settings.")
