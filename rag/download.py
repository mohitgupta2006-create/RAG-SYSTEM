"""Size-limited downloads pinned to a validated public destination address."""

import ipaddress
import socket
import time
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit

import certifi
import urllib3

from rag.errors import DownloadError


def _destination(url: str, allowed_hosts: tuple[str, ...] = ()) -> tuple:
    try:
        parsed = urlsplit(url)
        host = (parsed.hostname or "").encode("idna").decode("ascii").lower().rstrip(".")
        if parsed.scheme not in {"http", "https"} or not host:
            raise ValueError("Use an HTTP(S) document URL")
        if parsed.username or parsed.password or parsed.fragment:
            raise ValueError("Credentials and fragments are not allowed")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if port != (443 if parsed.scheme == "https" else 80):
            raise ValueError("Only default HTTP(S) ports are allowed")
        if allowed_hosts and host not in allowed_hosts:
            raise ValueError("Document host is not allowed")
        addresses = {item[4][0] for item in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)}
        if not addresses or any(
            not ipaddress.ip_address(address).is_global for address in addresses
        ):
            raise ValueError("Document host must resolve exclusively to public IP addresses")
        return parsed, host, port, min(addresses)
    except (ValueError, UnicodeError, OSError) as exc:
        raise DownloadError() from exc


def source_filename(url: str) -> str:
    # Strip query parameters (which may contain signed access tokens).
    name = unquote(urlsplit(url).path).replace("\\", "/").rsplit("/", 1)[-1]
    name = "".join(c for c in name if c.isalnum() or c in " ._-()").strip(" .")
    return name[:150] or "document.pdf"


def download_pdf(
    url: str,
    destination: Path,
    *,
    max_bytes: int,
    timeout: float,
    allowed_hosts: tuple[str, ...] = (),
    max_redirects: int = 3,
) -> None:
    deadline = time.monotonic() + timeout
    try:
        for hop in range(max_redirects + 1):
            parsed, host, port, address = _destination(url, allowed_hosts)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise DownloadError()
            kwargs = dict(host=address, port=port, maxsize=1)
            if parsed.scheme == "https":
                pool = urllib3.HTTPSConnectionPool(
                    **kwargs,
                    assert_hostname=host,
                    server_hostname=host,
                    cert_reqs="CERT_REQUIRED",
                    ca_certs=certifi.where(),
                )
            else:
                pool = urllib3.HTTPConnectionPool(**kwargs)
            try:
                target = parsed.path or "/"
                if parsed.query:
                    target += "?" + parsed.query
                response = pool.urlopen(
                    "GET",
                    target,
                    headers={
                        "Host": f"[{host}]" if ":" in host else host,
                        "Accept-Encoding": "identity",
                    },
                    timeout=urllib3.Timeout(connect=min(10, remaining), read=remaining),
                    retries=False,
                    redirect=False,
                    preload_content=False,
                )
                try:
                    if response.status in {301, 302, 303, 307, 308}:
                        if hop == max_redirects or not response.headers.get("Location"):
                            raise DownloadError()
                        next_url = urljoin(url, response.headers["Location"])
                        if parsed.scheme == "https" and urlsplit(next_url).scheme != "https":
                            raise DownloadError()
                        url = next_url
                        continue
                    if response.status != 200:
                        raise DownloadError()
                    if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                        raise DownloadError()
                    length = response.headers.get("Content-Length")
                    if length and (int(length) < 0 or int(length) > max_bytes):
                        raise DownloadError()
                    size = 0
                    prefix = bytearray()
                    with destination.open("wb") as output:
                        for block in response.stream(8192, decode_content=False):
                            size += len(block)
                            if size > max_bytes or time.monotonic() > deadline:
                                raise DownloadError()
                            if len(prefix) < 1024:
                                prefix.extend(block[: 1024 - len(prefix)])
                            output.write(block)
                    if b"%PDF-" not in prefix:
                        raise DownloadError()
                    return
                finally:
                    response.close()
            finally:
                pool.close()
        raise DownloadError()
    except (DownloadError, OSError, ValueError, urllib3.exceptions.HTTPError) as exc:
        destination.unlink(missing_ok=True)
        raise DownloadError() from exc
