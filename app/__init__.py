from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.proxy_headers import ProxyHeadersMiddleware

from .database import init_db
from .routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):  # pragma: no cover
    init_db()
    yield


def create_app() -> FastAPI:
    """Application factory for the tournament bracket site."""
    base_dir = Path(__file__).resolve().parent
    app = FastAPI(title="Freeze Fest Bracket Builder", lifespan=lifespan)
    app.add_middleware(ProxyHeadersMiddleware, trusted_hosts=["*"])
    app.include_router(router)
    static_dir = base_dir / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    return app


app = create_app()
