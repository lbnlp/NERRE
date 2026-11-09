"""Public API shapes for sentence-level extraction."""
from typing import Literal

from pydantic import BaseModel, Field


class ExtractedSentence(BaseModel):
    sentence: str
    status: Literal["complete"]
    basemats: dict[str, str]
    dopants: dict[str, str]
    dopants2basemats: dict[str, list[str]]


class FailedSentence(BaseModel):
    sentence: str
    status: Literal["error"]
    error: str


class ExtractionResponse(BaseModel):
    filename: str
    schema_name: Literal["doping"] = Field(alias="schema")
    status: Literal["complete", "partial"]
    sentences: list[ExtractedSentence | FailedSentence]
