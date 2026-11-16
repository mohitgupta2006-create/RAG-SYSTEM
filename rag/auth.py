"""Bearer authentication that does not disclose configured credentials."""

import secrets


def valid_api_key(provided: str, expected: str) -> bool:
    return secrets.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))
