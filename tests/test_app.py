import os
import tempfile
from pathlib import Path

import pytest
import pytest_asyncio
import httpx
from sqlmodel import Session, select


TEST_DB = Path(tempfile.gettempdir()) / "freeze_fest_test_app.sqlite3"
os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB}"

from app import app  # noqa: E402
from app.database import Event, engine, init_db  # noqa: E402


@pytest.fixture(autouse=True)
def _cleanup_db():
    if TEST_DB.exists():
        TEST_DB.unlink()
    init_db()
    yield
    engine.dispose()
    if TEST_DB.exists():
        TEST_DB.unlink()


@pytest_asyncio.fixture
async def async_client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


@pytest.mark.asyncio
async def test_index_returns_200(async_client):
    response = await async_client.get("/")
    assert response.status_code == 200
    assert "Freeze Fest" in response.text


def test_events_seeded():
    with Session(engine) as session:
        events = session.exec(select(Event)).all()
    slugs = {event.slug for event in events}
    assert "2025" in slugs
    assert "2024" in slugs
