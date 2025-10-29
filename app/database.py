"""Database models and helpers for event, team, and media management."""

from __future__ import annotations

import os
import shutil
from datetime import date, datetime
from pathlib import Path
from typing import Iterator

from sqlmodel import Field, Session, SQLModel, create_engine, select

BASE_DIR = Path(__file__).resolve().parent


DEFAULT_SQLITE_PATH = "sqlite:///./freeze_fest.db"
ACTIVE_EVENT_SLUG = "freeze-fest-2025"
EVENT_DEFINITIONS = [
    {
        "name": "Freeze Fest 2025",
        "slug": ACTIVE_EVENT_SLUG,
        "description": "Annual cornhole, bucket golf, and kanban showdown",
        "location": "South Valley, Albuquerque, NM",
        "event_date": date(2025, 11, 15),
        "games": "Cornhole, Bucket Golf, Kanban",
        "winners": None,
    },
    {
        "name": "Freeze Fest 2024",
        "slug": "freeze-fest-2024",
        "description": "Tri-game series celebrating the end of mosquito season",
        "location": "South Valley, Albuquerque, NM",
        "event_date": date(2024, 11, 11),
        "games": "Cornhole, Kanban, Rollors",
        "winners": "John & Stefan",
    },
]


def _build_engine_url() -> str:
    """Return the configured database URL or fall back to SQLite."""
    return os.environ.get("DATABASE_URL", DEFAULT_SQLITE_PATH)


def _build_engine() -> "Engine":
    url = _build_engine_url()
    engine_kwargs = {}
    if url.startswith("sqlite"):
        # SQLite needs check_same_thread disabled for FastAPI concurrency,
        # but passing this flag to other drivers (e.g., psycopg2) raises errors.
        engine_kwargs["connect_args"] = {"check_same_thread": False}
    return create_engine(url, **engine_kwargs)


engine = _build_engine()

STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"
SAMPLE_IMAGE = STATIC_DIR / "img/freezefest.png"
SAMPLE_UPLOAD_NAME = "sample-freezefest.png"

ACTIVE_EVENT_DEFINITION = {
    "name": "Freeze Fest 2025",
    "slug": "freeze-fest-2025",
    "description": "Annual cornhole, bucket golf, and kanban showdown",
    "location": "South Valley, Albuquerque, NM",
    "event_date": date(2025, 11, 15),
}


class Event(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(nullable=False)
    slug: str = Field(nullable=False, unique=True, index=True)
    description: str | None = Field(default=None)
    location: str | None = Field(default=None)
    event_date: date = Field(nullable=False)
    games: str = Field(default="Cornhole, Bucket Golf, Kanban", nullable=False)
    winners: str | None = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)


class Team(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True, min_length=2, max_length=128, nullable=False, unique=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    event_id: int = Field(foreign_key="event.id", nullable=False, index=True)


class Photo(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    filename: str = Field(nullable=False, unique=True)
    original_name: str = Field(default="Upload", nullable=False)
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    event_id: int = Field(foreign_key="event.id", nullable=False, index=True)


class RSVP(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(nullable=False, max_length=120)
    email: str = Field(nullable=False, max_length=255)
    guests: int = Field(default=1, nullable=False)
    message: str | None = Field(default=None)
    event_id: int = Field(foreign_key="event.id", nullable=False, index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)


class FreeAgent(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(nullable=False, max_length=120)
    email: str = Field(nullable=False, max_length=255)
    note: str | None = Field(default=None)
    status: str = Field(default="pending", nullable=False)
    event_id: int = Field(foreign_key="event.id", nullable=False, index=True)
    team_id: int | None = Field(default=None, foreign_key="team.id")
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)


class Match(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    event_id: int = Field(foreign_key="event.id", nullable=False, index=True)
    game: str = Field(nullable=False)
    order_index: int = Field(nullable=False, index=True)
    team1_id: int = Field(foreign_key="team.id", nullable=False)
    team2_id: int = Field(foreign_key="team.id", nullable=False)
    score_team1: int | None = Field(default=None)
    score_team2: int | None = Field(default=None)
    status: str = Field(default="pending", nullable=False)
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)


def init_db() -> None:
    """Create tables if they don't already exist."""
    SQLModel.metadata.create_all(engine)
    _ensure_upload_dir()
    events = _ensure_events()
    _seed_sample_photos(events)


def get_session() -> Iterator[Session]:
    """Yield a SQLModel session for dependency injection."""
    with Session(engine) as session:
        yield session


def _ensure_upload_dir() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _ensure_events() -> list[Event]:
    ensured: list[Event] = []
    with Session(engine, expire_on_commit=False) as session:
        for definition in EVENT_DEFINITIONS:
            event = session.exec(select(Event).where(Event.slug == definition["slug"])).first()
            if not event:
                event = Event(**definition)
                session.add(event)
                session.commit()
                session.refresh(event)
            ensured.append(event)
        for event in ensured:
            session.expunge(event)
    return ensured


def get_active_event(session: Session) -> Event:
    event = session.exec(select(Event).where(Event.slug == ACTIVE_EVENT_SLUG)).first()
    if event:
        return event
    raise RuntimeError("Active event not found. Verify EVENT_DEFINITIONS and database state.")


def _seed_sample_photos(events: list[Event]) -> None:
    if not SAMPLE_IMAGE.exists():
        return

    with Session(engine) as session:
        for event in events:
            sample_filename = f"{event.slug}-sample.png"
            sample_target = UPLOAD_DIR / sample_filename
            if not sample_target.exists():
                shutil.copyfile(SAMPLE_IMAGE, sample_target)

            existing = session.exec(select(Photo).where(Photo.event_id == event.id).limit(1)).first()
            if existing:
                continue
            session.add(
                Photo(
                    filename=sample_filename,
                    original_name=f"{event.name} Sample",
                    event_id=event.id,
                )
            )
            session.commit()
