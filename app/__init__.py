from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .database import init_db
from .routes import router


def create_app() -> FastAPI:
    """Application factory for the tournament bracket site."""
    app = FastAPI(title="Freeze Fest Bracket Builder")
    app.include_router(router)
    app.mount("/static", StaticFiles(directory="app/static"), name="static")

    # Ensure database tables exist once the application starts.
    @app.on_event("startup")
    async def _startup() -> None:  # pragma: no cover - lifecycle hook
        init_db()

    return app


app = create_app()
