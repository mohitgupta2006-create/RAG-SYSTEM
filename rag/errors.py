"""Errors safe to translate into public API responses."""


class RAGError(Exception):
    status_code = 500
    public_message = "The document could not be processed."


class DocumentError(RAGError):
    status_code = 422
    public_message = "The PDF is invalid, encrypted, empty, or has no extractable text."


class DownloadError(RAGError):
    status_code = 400
    public_message = "The document URL could not be downloaded."


class GenerationError(RAGError):
    status_code = 502
    public_message = "The answer provider could not return a valid grounded answer."


class CacheError(RAGError):
    status_code = 503
    public_message = "The document cache is unavailable. Please retry."


class BusyError(RAGError):
    status_code = 503
    public_message = "The service is busy. Please retry shortly."
