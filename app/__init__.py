from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .database import init_db
from .routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):  # pragma: no cover
    try:
        import os
        print("Cloud SQL dir contents:", os.listdir("/cloudsql"))
    except FileNotFoundError:
        print("Cloud SQL dir missing; /cloudsql not found")
    init_db()
    yield


def create_app() -> FastAPI:
    """Application factory for the tournament bracket site."""
    base_dir = Path(__file__).resolve().parent
    app = FastAPI(title="Freeze Fest Bracket Builder", lifespan=lifespan)
    app.include_router(router)
    static_dir = base_dir / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    return app


app = create_app()
