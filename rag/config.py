"""Validated runtime configuration; importing this module has no side effects."""

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)

    api_key: SecretStr
    gemini_api_key: SecretStr
    gemini_model: str = Field(min_length=1)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    cache_dir: Path = Path("knowledge_base_cache")
    max_document_bytes: int = Field(default=25 * 1024 * 1024, ge=1024)
    max_pages: int = Field(default=500, ge=1, le=5000)
    download_timeout: float = Field(default=30, gt=0, le=300)
    generation_timeout: float = Field(default=60, gt=0, le=300)
    max_cached_documents: int = Field(default=8, ge=1, le=100)
    max_concurrent_requests: int = Field(default=2, ge=1, le=32)
    allowed_document_hosts: tuple[str, ...] = ()

    @field_validator("api_key", "gemini_api_key")
    @classmethod
    def reject_placeholder_keys(cls, value: SecretStr) -> SecretStr:
        raw = value.get_secret_value().strip()
        if not raw or raw.startswith("your_"):
            raise ValueError("Set a real API key instead of a placeholder")
        return SecretStr(raw)

    @field_validator("gemini_model", "embedding_model")
    @classmethod
    def strip_model(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Model name cannot be blank")
        return value.strip()

    @classmethod
    def from_env(cls) -> "Settings":
        # Read .env only when explicitly creating runtime settings.
        from dotenv import load_dotenv

        load_dotenv()
        values = {}
        for name in cls.model_fields:
            raw = os.getenv(name.upper())
            if raw is not None:
                values[name] = raw
        if "allowed_document_hosts" in values:
            values["allowed_document_hosts"] = tuple(
                host.strip().lower() for host in values["allowed_document_hosts"].split(",")
                if host.strip()
            )
        return cls(**values)
