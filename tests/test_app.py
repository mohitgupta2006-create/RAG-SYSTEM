import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app import create_app
from rag.config import Settings
from rag.errors import BusyError, GenerationError


@pytest.fixture
def service():
    value = Mock()
    value.answer.return_value = ["An answer"]
    return value


@pytest.fixture
def client(service, tmp_path):
    config = Settings(
        api_key="test-token",
        gemini_api_key="provider-token",
        gemini_model="test-model",
        cache_dir=tmp_path,
    )
    with TestClient(create_app(config, lambda _: service), raise_server_exceptions=False) as value:
        yield value
    service.close.assert_called_once()


PAYLOAD = dict(document="https://example.com/policy.pdf", queries=["Waiting period?"])
HEADERS = {"Authorization": "Bearer test-token"}


@pytest.mark.parametrize("path", ["/api/run", "/hackrx/run"])
def test_both_routes_and_request_id(client, path):
    response = client.post(path, headers=HEADERS, json=PAYLOAD)
    assert response.status_code == 200
    assert response.json() == {"answers": ["An answer"]}
    assert len(response.headers["x-request-id"]) == 32


def test_auth_and_validation(client, service):
    assert client.post("/api/run", json=PAYLOAD).status_code == 401
    assert (
        client.post("/api/run", headers={"Authorization": "Bearer wrong"}, json=PAYLOAD).status_code
        == 401
    )
    assert (
        client.post("/api/run", headers=HEADERS, json={**PAYLOAD, "queries": []}).status_code == 422
    )
    service.answer.assert_not_called()


def test_request_body_is_bounded(client, service):
    response = client.post("/api/run", headers=HEADERS, content=b"x" * (256 * 1024 + 1))
    assert response.status_code == 413
    service.answer.assert_not_called()


def test_typed_and_unexpected_errors_do_not_leak_details(client, service):
    service.answer.side_effect = GenerationError("secret provider details")
    response = client.post("/api/run", headers=HEADERS, json=PAYLOAD)
    assert response.status_code == 502
    assert "secret" not in response.text
    service.answer.side_effect = RuntimeError("secret traceback")
    response = client.post("/api/run", headers=HEADERS, json=PAYLOAD)
    assert response.status_code == 500
    assert "secret" not in response.text
    service.answer.side_effect = BusyError()
    response = client.post("/api/run", headers=HEADERS, json=PAYLOAD)
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "10"


def test_health_remains_responsive_during_blocking_work(client, service):
    entered, release = threading.Event(), threading.Event()

    def blocked(*args):
        entered.set()
        assert release.wait(5)
        return ["answer"]

    service.answer.side_effect = blocked
    with ThreadPoolExecutor() as pool:
        pending = pool.submit(client.post, "/api/run", headers=HEADERS, json=PAYLOAD)
        try:
            assert entered.wait(5)
            assert client.get("/health").status_code == 200
            assert client.get("/ready").status_code == 200
        finally:
            release.set()
        assert pending.result(timeout=5).status_code == 200


def test_create_app_does_not_read_credentials_or_load_models(monkeypatch):
    def unexpected_configuration():
        raise AssertionError("Import and app construction must not load settings")

    monkeypatch.setattr(Settings, "from_env", unexpected_configuration)
    application = create_app()
    assert application.title == "Insurance Policy RAG API"
    assert not hasattr(application.state, "service")
