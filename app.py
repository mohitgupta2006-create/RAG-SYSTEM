"""FastAPI entry point. Models and credentials are loaded only during startup."""

import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.concurrency import run_in_threadpool

from rag.auth import valid_api_key
from rag.config import Settings
from rag.errors import BusyError, RAGError
from rag.middleware import RequestMiddleware
from rag.schemas import EndpointRequest, EndpointResponse, HealthResponse

logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)


def create_app(settings: Settings | None = None, service_factory=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application):
        config = settings or Settings.from_env()
        factory = service_factory
        if factory is None:
            from rag.service import RAGService

            factory = RAGService
        application.state.settings = config
        application.state.service = await run_in_threadpool(factory, config)
        application.state.ready = True
        try:
            yield
        finally:
            application.state.ready = False
            await run_in_threadpool(application.state.service.close)

    application = FastAPI(title="Insurance Policy RAG API", version="2.0.0", lifespan=lifespan)
    application.add_middleware(RequestMiddleware)

    @application.exception_handler(RAGError)
    async def service_error(request, exc):
        logger.warning("Service request failed (%s)", type(exc).__name__)
        headers = {"Retry-After": "10"} if isinstance(exc, BusyError) else None
        return JSONResponse({"detail": exc.public_message}, exc.status_code, headers=headers)

    @application.exception_handler(Exception)
    async def unexpected_error(request, exc):
        logger.error("Unexpected service failure (%s)", type(exc).__name__)
        return JSONResponse({"detail": "An internal server error occurred."}, 500)

    async def verify_api_key(
        request: Request,
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
    ):
        expected = request.app.state.settings.api_key.get_secret_value()
        if credentials is None or not valid_api_key(credentials.credentials, expected):
            raise HTTPException(
                401, "Invalid or missing API key.", headers={"WWW-Authenticate": "Bearer"}
            )

    @application.get("/")
    def root():
        return {"message": "Insurance Policy RAG API", "docs": "/docs"}

    @application.get("/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(status="Healthy", timestamp=datetime.now(UTC).isoformat())

    @application.get("/ready")
    async def ready(request: Request):
        if not getattr(request.app.state, "ready", False):
            raise HTTPException(503, "Runtime is not ready.")
        return {"status": "Ready"}

    @application.post(
        "/api/run", response_model=EndpointResponse, dependencies=[Depends(verify_api_key)]
    )
    @application.post(
        "/hackrx/run", response_model=EndpointResponse, dependencies=[Depends(verify_api_key)]
    )
    def generate_response(payload: EndpointRequest, request: Request):
        # FastAPI runs sync handlers in its worker pool, leaving the event loop responsive.
        answers = request.app.state.service.answer(str(payload.document), payload.queries)
        return EndpointResponse(answers=answers)

    return application


app = create_app()


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="127.0.0.1", port=8000)
