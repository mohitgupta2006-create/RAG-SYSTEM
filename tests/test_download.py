from unittest.mock import Mock

import pytest

from rag.download import _destination, download_pdf, source_filename
from rag.errors import DownloadError


def dns(monkeypatch, address="93.184.216.34"):
    monkeypatch.setattr(
        "rag.download.socket.getaddrinfo", lambda *a, **k: [(2, 1, 6, "", (address, 443))]
    )


@pytest.mark.parametrize(
    "address", ["127.0.0.1", "10.0.0.1", "169.254.169.254", "::1", "192.168.1.1"]
)
def test_private_destinations_are_rejected(monkeypatch, address):
    dns(monkeypatch, address)
    with pytest.raises(DownloadError):
        _destination("https://example.com/a.pdf")


def test_pool_is_pinned_to_checked_ip(monkeypatch, tmp_path):
    dns(monkeypatch)
    response = Mock(status=200, headers={})
    response.stream.return_value = [b"%PDF-1.7\ncontent"]
    factory = Mock()
    factory.return_value.urlopen.return_value = response
    monkeypatch.setattr("rag.download.urllib3.HTTPSConnectionPool", factory)
    path = tmp_path / "download.pdf"
    download_pdf("https://example.com/a.pdf", path, max_bytes=1024, timeout=5)
    assert factory.call_args.kwargs["host"] == "93.184.216.34"
    assert factory.call_args.kwargs["server_hostname"] == "example.com"
    assert path.read_bytes().startswith(b"%PDF-")
    response.close.assert_called_once()


@pytest.mark.parametrize("blocks", [[b"not a PDF"], [b"%PDF-" + b"x" * 100]])
def test_invalid_or_oversized_download_is_cleaned(monkeypatch, tmp_path, blocks):
    dns(monkeypatch)
    response = Mock(status=200, headers={})
    response.stream.return_value = blocks
    pool = Mock()
    pool.urlopen.return_value = response
    monkeypatch.setattr("rag.download.urllib3.HTTPSConnectionPool", lambda **k: pool)
    path = tmp_path / "partial.pdf"
    with pytest.raises(DownloadError):
        download_pdf("https://example.com/a.pdf", path, max_bytes=30, timeout=5)
    assert not path.exists()


def test_redirect_to_private_network_is_blocked(monkeypatch, tmp_path):
    def resolve(host, *a, **k):
        return [(2, 1, 6, "", ("127.0.0.1" if host == "internal" else "93.184.216.34", 443))]

    monkeypatch.setattr("rag.download.socket.getaddrinfo", resolve)
    response = Mock(status=302, headers={"Location": "https://internal/a.pdf"})
    pool = Mock()
    pool.urlopen.return_value = response
    monkeypatch.setattr("rag.download.urllib3.HTTPSConnectionPool", lambda **k: pool)
    with pytest.raises(DownloadError):
        download_pdf("https://example.com/a.pdf", tmp_path / "a.pdf", max_bytes=1024, timeout=5)
    assert pool.urlopen.call_count == 1


def test_allowlist_and_filename(monkeypatch):
    dns(monkeypatch)
    with pytest.raises(DownloadError):
        _destination("https://other.example/a.pdf", ("trusted.example",))
    assert (
        source_filename("https://example.com/Policy%20Terms.pdf?secret=token") == "Policy Terms.pdf"
    )
