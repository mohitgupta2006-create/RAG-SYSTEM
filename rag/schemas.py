"""Bounded API inputs and compatible string-answer responses."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, StringConstraints, field_validator

Query = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


class EndpointRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document: HttpUrl = Field(description="Public HTTP(S) URL of a PDF")
    queries: list[Query] = Field(min_length=1, max_length=20)

    @field_validator("document")
    @classmethod
    def validate_document(cls, value: HttpUrl) -> HttpUrl:
        if value.username or value.password or value.fragment:
            raise ValueError("Document URL cannot contain credentials or a fragment")
        if len(str(value)) > 8192:
            raise ValueError("Document URL is too long")
        return value


class EndpointResponse(BaseModel):
    answers: list[str]


class HealthResponse(BaseModel):
    status: str
    timestamp: str
