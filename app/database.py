"""Database models and helpers for event, team, and media management."""

from __future__ import annotations

import os
import shutil
import logging
from datetime import date, datetime
from io import BytesIO
from pathlib import Path
from typing import Iterator

from sqlmodel import Field, Session, SQLModel, create_engine, select

try:  # Optional conversion support
    from pillow_heif import read_heif
    from PIL import Image
except ImportError:  # pragma: no cover
    read_heif = None
    Image = None

BASE_DIR = Path(__file__).resolve().parent


DEFAULT_SQLITE_PATH = "sqlite:///./freeze_fest.db"
ACTIVE_EVENT_SLUG = "2025"
LOCAL_PHOTO_IMPORT_DIR = os.getenv("LOCAL_PHOTO_IMPORT_DIR_CONTAINER") or os.getenv("LOCAL_PHOTO_IMPORT_DIR")
LOCAL_PHOTO_IMPORT_EVENT_SLUG = os.getenv("LOCAL_PHOTO_IMPORT_EVENT_SLUG")

logger = logging.getLogger(__name__)
EVENT_DEFINITIONS = [
    {
        "name": "Freeze Fest 2025",
        "slug": ACTIVE_EVENT_SLUG,
        "description": "Annual cornhole, bucket golf, and KanJam showdown",
        "location": "South Valley, Albuquerque, NM",
        "event_date": date(2025, 11, 15),
        "games": "Cornhole, Bucket Golf, KanJam",
        "winners": None,
        "winner_photo": None,
    },
    {
        "name": "Freeze Fest 2024",
        "slug": "2024",
        "description": (
            "Freeze Fest 2024 was a beautiful sunny day. It began with Oompa Do yelling, "
            "\"PLEASE DON'T BURN LEAVES TODAY!\" across the neighbors' yards while Dan and "
            "Maely looked on—confused and slightly embarrassed. Friends trickled in, enchiladas "
            "hit the tables, and the good company set the tone for the afternoon.\n\n"
            "Once play started, the organizers quickly realized the paper brackets couldn't keep "
            "up with the action. The scramble to coordinate matches is what ultimately inspired "
            "the Freeze Fest website.\n\nAfter group play, controversy struck. One team dominated "
            "every game, yet a championship round hadn't been clearly announced. A three-team "
            "playoff settled the debate, John and Stefan advanced, and they went on to win the "
            "final. As the sun dipped too quickly, the evening closed with stories about making "
            "tortillas around the fire."
        ),
        "location": "South Valley, Albuquerque, NM",
        "event_date": date(2024, 11, 16),
        "games": "Cornhole, KanJam, Rollors",
        "winners": "John & Stefan",
        "winner_photo": None,
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
    "name": "Freeze Fest " + ACTIVE_EVENT_SLUG,
    "slug": ACTIVE_EVENT_SLUG,
    "description": "Annual cornhole, bucket golf, and KanJam showdown",
    "location": "South Valley, Albuquerque, NM",
    "event_date": date(2025, 11, 15),
    "winner_photo": None,
}


class Event(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(nullable=False)
    slug: str = Field(nullable=False, unique=True, index=True)
    description: str | None = Field(default=None)
    location: str | None = Field(default=None)
    event_date: date = Field(nullable=False)
    games: str = Field(default="Cornhole, Bucket Golf, KanJam", nullable=False)
    winners: str | None = Field(default=None)
    winner_photo: str | None = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)


class Team(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True, min_length=2, max_length=128, nullable=False, unique=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    event_id: int = Field(foreign_key="event.id", nullable=False, index=True)
    member_one: str | None = Field(default=None, max_length=128)
    member_two: str | None = Field(default=None, max_length=128)


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
    is_playoff: bool = Field(default=False, nullable=False)
    playoff_round: str | None = Field(default=None, max_length=50)
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)


def init_db() -> None:
    """Create tables if they don't already exist."""
    SQLModel.metadata.create_all(engine)
    _ensure_team_member_columns()
    _ensure_playoff_columns()
    _ensure_event_winner_photo_column()
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

        _import_local_photos(session, events)


def _import_local_photos(session: Session, events: list[Event]) -> None:
    if not LOCAL_PHOTO_IMPORT_DIR or not LOCAL_PHOTO_IMPORT_EVENT_SLUG:
        return

    source_dir = Path(LOCAL_PHOTO_IMPORT_DIR)
    if not source_dir.exists():
        logger.warning("Local photo import directory %s not found", source_dir)
        return

    target_event = next((event for event in events if event.slug == LOCAL_PHOTO_IMPORT_EVENT_SLUG), None)
    if not target_event:
        logger.warning("No event matches LOCAL_PHOTO_IMPORT_EVENT_SLUG=%s", LOCAL_PHOTO_IMPORT_EVENT_SLUG)
        return

    allowed_suffixes = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".heic", ".heif"}
    existing = {
        photo.filename
        for photo in session.exec(select(Photo).where(Photo.event_id == target_event.id))
    }

    for file_path in sorted(source_dir.iterdir()):
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()
        if suffix not in allowed_suffixes:
            continue

        dest_suffix = suffix
        data_bytes: bytes | None = None
        try:
            if suffix in {".heic", ".heif"}:
                if not read_heif or not Image:
                    logger.warning("Skipping %s: HEIC support unavailable", file_path.name)
                    continue
                heif_file = read_heif(file_path)
                img = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data, "raw")
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                data_bytes = buffer.getvalue()
                dest_suffix = ".jpg"
            else:
                data_bytes = file_path.read_bytes()
        except Exception as exc:  # pragma: no cover - best effort import
            logger.warning("Skipping %s: %s", file_path.name, exc)
            continue

        if data_bytes is None:
            continue

        dest_stem = file_path.stem
        dest_name = f"{dest_stem}{dest_suffix}"
        dest_path = UPLOAD_DIR / dest_name
        counter = 1
        while dest_path.exists():
            dest_name = f"{dest_stem}_{counter}{dest_suffix}"
            dest_path = UPLOAD_DIR / dest_name
            counter += 1

        dest_path.write_bytes(data_bytes)

        if dest_name in existing:
            continue

        session.add(
            Photo(
                filename=dest_name,
                original_name=file_path.name,
                event_id=target_event.id,
            )
        )
        existing.add(dest_name)

    session.commit()


def _ensure_team_member_columns() -> None:
    with engine.begin() as conn:
        dialect = conn.dialect.name
        existing_columns: set[str] = set()
        if dialect == "sqlite":
            rows = conn.exec_driver_sql("PRAGMA table_info('team')")
            existing_columns = {row[1] for row in rows}
        else:
            rows = conn.exec_driver_sql(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name='team' AND table_schema = current_schema()"
            )
            existing_columns = {row[0] for row in rows}

        statements: list[str] = []
        if "member_one" not in existing_columns:
            statements.append("ALTER TABLE team ADD COLUMN member_one VARCHAR(128)")
        if "member_two" not in existing_columns:
            statements.append("ALTER TABLE team ADD COLUMN member_two VARCHAR(128)")

        for stmt in statements:
            conn.exec_driver_sql(stmt)


def _ensure_playoff_columns() -> None:
    with engine.begin() as conn:
        dialect = conn.dialect.name
        if dialect == "sqlite":
            rows = conn.exec_driver_sql("PRAGMA table_info('match')")
            existing_columns = {row[1] for row in rows}
        else:
            rows = conn.exec_driver_sql(
                "SELECT column_name FROM information_schema.columns WHERE table_name='match' AND table_schema = current_schema()"
            )
            existing_columns = {row[0] for row in rows}

        statements: list[str] = []
        if "is_playoff" not in existing_columns:
            if dialect == "sqlite":
                statements.append("ALTER TABLE match ADD COLUMN is_playoff INTEGER DEFAULT 0 NOT NULL")
            else:
                statements.append("ALTER TABLE match ADD COLUMN is_playoff BOOLEAN DEFAULT FALSE NOT NULL")
        if "playoff_round" not in existing_columns:
            column_type = "TEXT" if dialect == "sqlite" else "VARCHAR(50)"
            statements.append(f"ALTER TABLE match ADD COLUMN playoff_round {column_type}")

        for stmt in statements:
            conn.exec_driver_sql(stmt)


def _ensure_event_winner_photo_column() -> None:
    with engine.begin() as conn:
        dialect = conn.dialect.name
        if dialect == "sqlite":
            rows = conn.exec_driver_sql("PRAGMA table_info('event')")
            existing_columns = {row[1] for row in rows}
        else:
            rows = conn.exec_driver_sql(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name='event' AND table_schema = current_schema()"
            )
            existing_columns = {row[0] for row in rows}

        if "winner_photo" not in existing_columns:
            column_type = "TEXT" if dialect == "sqlite" else "VARCHAR(255)"
            conn.exec_driver_sql(f"ALTER TABLE event ADD COLUMN winner_photo {column_type}")
