"""Bound request bodies before JSON parsing and attach safe request diagnostics."""

import logging
import time
import uuid

from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


class RequestMiddleware:
    def __init__(self, app, max_body_bytes=256 * 1024):
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        identity = uuid.uuid4().hex
        start = time.monotonic()
        status_code = 500

        async def respond(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                message["headers"] = list(message.get("headers", [])) + [
                    (b"x-request-id", identity.encode())
                ]
            await send(message)

        bounded_receive = receive
        try:
            if scope["method"] == "POST":
                body = bytearray()
                while True:
                    event = await receive()
                    if event["type"] == "http.disconnect":
                        return
                    block = event.get("body", b"")
                    if len(body) + len(block) > self.max_body_bytes:
                        return await JSONResponse({"detail": "Request body exceeds 256 KiB."}, 413)(
                            scope, receive, respond
                        )
                    body.extend(block)
                    if not event.get("more_body", False):
                        break
                replayed = False

                async def replay():
                    nonlocal replayed
                    if not replayed:
                        replayed = True
                        return {"type": "http.request", "body": bytes(body), "more_body": False}
                    return await receive()

                bounded_receive = replay
            await self.app(scope, bounded_receive, respond)
        finally:
            logger.info(
                "request_id=%s method=%s status=%s duration_ms=%.1f",
                identity,
                scope["method"],
                status_code,
                (time.monotonic() - start) * 1000,
            )
