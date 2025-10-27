from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .database import init_db
from .routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):  # pragma: no cover
    init_db()
    yield


def create_app() -> FastAPI:
    """Application factory for the tournament bracket site."""
    app = FastAPI(title="Freeze Fest Bracket Builder", lifespan=lifespan)
    app.include_router(router)
    app.mount("/static", StaticFiles(directory="app/static"), name="static")
    return app


app = create_app()
