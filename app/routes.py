from __future__ import annotations

import copy
import json
import logging
import hmac
import itertools
import math
import os
import re
import secrets
import shutil
import smtplib
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
from uuid import uuid4
from urllib.parse import quote, urljoin

from collections import defaultdict

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlmodel import Session, SQLModel, select, func
from email.message import EmailMessage

try:  # Optional – used for HEIC conversion
    from pillow_heif import read_heif
    from PIL import Image
except ImportError:  # pragma: no cover - graceful fallback when libs missing
    read_heif = None
    Image = None

from .bracket import GAMES, generate_schedule

from .database import (
    ACTIVE_EVENT_SLUG,
    Event,
    FreeAgent,
    Match,
    Photo,
    RSVP,
    Team,
    UPLOAD_DIR,
    get_active_event,
    get_session,
    delete_pending_change,
    get_pending_change,
    upsert_pending_change,
    PendingChange,
)
from .storage import (
    extract_object_name,
    gcs_photos_enabled,
    gcs_public_url,
    is_gcs_identifier,
    make_gcs_identifier,
    upload_photo_stream,
)

BASE_DIR = Path(__file__).resolve().parent

router = APIRouter()
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

ALLOWED_FREE_AGENT_STATUSES = {"pending", "paired"}

MAX_OPEN_MATCHES_PER_GAME = {
    "Bucket Golf": 2,
    "Bucket Golf Semifinal": 1,
}

logger = logging.getLogger(__name__)
USE_GCS_PHOTOS = gcs_photos_enabled()
EVENT_GALLERY_PREVIEW_LIMIT = 12
CAST_PHOTO_LIMIT = 40
CAST_APP_ID = os.getenv("CAST_APP_ID")
CAST_SENDER_ENABLED = bool(CAST_APP_ID)
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL")
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT_RAW = os.getenv("SMTP_PORT")
SMTP_PORT = int(SMTP_PORT_RAW) if SMTP_PORT_RAW and SMTP_PORT_RAW.isdigit() else None
SMTP_USERNAME = os.getenv("SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "true").lower() in {"1", "true", "yes"}
SMTP_USE_SSL = os.getenv("SMTP_USE_SSL", "false").lower() in {"1", "true", "yes"}
SMTP_SENDER = os.getenv("SMTP_SENDER")

MODERATION_APPROVED = "approved"
MODERATION_BLOCKED = "blocked"
REVIEW_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"http[s]?://",
        r"www\.",
        r"<[^>]+>",
        r"\bseo\b",
        r"\bviagra\b",
    ]
]
MAX_TEXT_LENGTH = 100


def _ensure_upload_dir() -> None:
    if not UPLOAD_DIR.exists():
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


SCHEDULER_DEBUG: dict[str, object] = {}
SESSION_COOKIE_NAME = os.getenv("ADMIN_SESSION_COOKIE", "freeze_admin_session")
SESSION_SECRET = os.getenv("ADMIN_SESSION_SECRET") or os.getenv("SECRET_KEY") or secrets.token_hex(32)
SESSION_MAX_AGE = int(os.getenv("ADMIN_SESSION_MAX_AGE", "43200"))  # 12 hours default
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "freeze-fest")


def _sign_payload(payload: str) -> str:
    secret = SESSION_SECRET.encode("utf-8")
    return hmac.new(secret, payload.encode("utf-8"), "sha256").hexdigest()


def _encode_session() -> str:
    token = secrets.token_hex(16)
    timestamp = str(int(time.time()))
    payload = f"{token}|{timestamp}"
    signature = _sign_payload(payload)
    return f"{payload}|{signature}"


def _decode_session(raw: str) -> bool:
    try:
        token, timestamp, signature = raw.split("|")
    except ValueError:
        return False
    payload = f"{token}|{timestamp}"
    expected = _sign_payload(payload)
    if not hmac.compare_digest(expected, signature):
        return False
    try:
        issued_at = int(timestamp)
    except ValueError:
        return False
    if int(time.time()) - issued_at > SESSION_MAX_AGE:
        return False
    return True


def notify_admin(subject: str, body: str) -> None:
    if not ADMIN_EMAIL:
        logger.info("Admin email not configured. Skipping notification: %s", subject)
        logger.debug("Notification body: %s", body)
        return

    sender = SMTP_SENDER or SMTP_USERNAME or ADMIN_EMAIL
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = ADMIN_EMAIL
    message.set_content(body)

    if not SMTP_HOST:
        logger.info("SMTP host not configured. Logging notification instead: %s", subject)
        logger.debug("Notification body: %s", body)
        return

    try:
        if SMTP_USE_SSL:
            with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT or 465) as server:
                if SMTP_USERNAME and SMTP_PASSWORD:
                    server.login(SMTP_USERNAME, SMTP_PASSWORD)
                server.send_message(message)
        else:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT or 587) as server:
                if SMTP_USE_TLS:
                    server.starttls()
                if SMTP_USERNAME and SMTP_PASSWORD:
                    server.login(SMTP_USERNAME, SMTP_PASSWORD)
                server.send_message(message)
    except Exception as exc:  # pragma: no cover - best effort logging
        logger.warning("Failed to send admin notification '%s': %s", subject, exc)


def needs_review(*values: str) -> bool:
    combined = " ".join(filter(None, values)).strip()
    if not combined:
        return False
    lower = combined.lower()
    if len(lower) > MAX_TEXT_LENGTH:
        return True
    for pattern in REVIEW_PATTERNS:
        if pattern.search(lower):
            return True
    return False


def _is_admin(request: Request) -> bool:
    cookie_value = request.cookies.get(SESSION_COOKIE_NAME)
    return bool(cookie_value and _decode_session(cookie_value))


def _admin_redirect(request: Request) -> RedirectResponse:
    next_path = request.url.path
    if request.url.query:
        next_path += f"?{request.url.query}"
    login_url = request.url_for("admin_login")
    redirect_target = f"{login_url}?next={quote(next_path, safe='')}"
    return RedirectResponse(redirect_target, status_code=303)


def _sanitize_next(next_param: str | None) -> str:
    if next_param and next_param.startswith("/"):
        return next_param
    return "/bracket"


def _absolute_next(request: Request, next_param: str) -> str:
    return urljoin(str(request.base_url), next_param.lstrip("/"))


def _team_display_name(session: Session, team_identifier: int | None) -> str:
    if not team_identifier:
        return "-- No team --"
    team = session.get(Team, team_identifier)
    return team.name if team else f"Team #{team_identifier}"


def _pending_updates(
    session: Session,
    entity_type: str,
    model_cls: type[SQLModel],
) -> list[dict[str, object]]:
    changes = session.exec(
        select(PendingChange)
        .where(PendingChange.entity_type == entity_type)
        .order_by(PendingChange.created_at.desc())
    ).all()
    if not changes:
        return []
    ids = {change.entity_id for change in changes}
    records = session.exec(select(model_cls).where(model_cls.id.in_(ids))).all()
    record_lookup = {record.id: record for record in records}
    payload: list[dict[str, object]] = []
    for change in changes:
        record = record_lookup.get(change.entity_id)
        if not record:
            continue
        payload.append(
            {
                "record": record,
                "proposed": json.loads(change.proposed_data),
                "original": json.loads(change.original_data),
                "submitted_at": change.updated_at,
            }
        )
    return payload


def _render(
    request: Request,
    template_name: str,
    context: dict[str, object],
    *,
    status_code: int | None = None,
) -> HTMLResponse:
    payload = dict(context)
    payload.setdefault("cast_app_id", CAST_APP_ID)
    payload.setdefault("cast_sender_enabled", CAST_SENDER_ENABLED)
    payload["is_admin"] = _is_admin(request)
    response = templates.TemplateResponse(request, template_name, payload)
    if status_code is not None:
        response.status_code = status_code
    return response


@router.get("/", response_class=HTMLResponse, name="index")
async def index(request: Request, session: Session = Depends(get_session)):
    status = request.query_params.get("rsvp")
    is_admin = _is_admin(request)
    context = _home_context(session, rsvp_status=status, include_blocked=is_admin)
    return _render(request, "index.html", context)


@router.get("/bracket", response_class=HTMLResponse, name="bracket")
async def bracket_page(request: Request, session: Session = Depends(get_session)):
    context = _schedule_context(session)
    status = request.query_params.get("tournament")
    if not status:
        status = request.query_params.get("playoff")
    messages = {
        "started": ("Tournament bracket generated. Good luck!", False),
        "needs-teams": ("At least two teams are required before starting the tournament.", True),
        "reset": ("All match scores have been cleared.", False),
        "not-started": ("No matches to reset yet—start the tournament first.", True),
        "score-locked": ("Only admins can change a score once it's been recorded.", True),
        "playoff-created": ("Playoff final created. Report scores when ready.", False),
        "playoff-need-scores": ("Bucket golf scores aren’t ready yet—finish group play first.", True),
        "playoff-tiebreak": ("Tie detected for the final slot. Run a tie-breaker before creating the playoff match.", True),
        "playoff-group-incomplete": ("Finish every round-robin match before starting the semifinal round.", True),
        "semis-created": ("Playoff semifinal round created. Post the new bucket golf scores below.", False),
        "semis-pending": ("Finish every semifinal score before creating the final.", True),
        "semis-need-teams": ("Need at least three teams with results before starting the semifinals.", True),
    }
    if status in messages:
        message, is_error = messages[status]
        context.update({"tournament_message": message, "tournament_error": is_error})
    return _render(request, "bracket.html", context)


@router.get("/admin/login", response_class=HTMLResponse, name="admin_login")
async def admin_login(request: Request, next: str | None = None):
    next_raw = _sanitize_next(next or request.query_params.get("next"))
    if _is_admin(request):
        return RedirectResponse(_absolute_next(request, next_raw), status_code=303)
    context = {"next": next_raw, "error": request.query_params.get("error")}
    return _render(request, "admin_login.html", context)


@router.post("/admin/login", response_class=HTMLResponse, name="admin_login_submit")
async def admin_login_submit(
    request: Request,
    username: str = Form(..., max_length=MAX_TEXT_LENGTH),
    password: str = Form(..., max_length=MAX_TEXT_LENGTH),
    next: str = Form(default="/bracket", max_length=MAX_TEXT_LENGTH),
):
    next_raw = _sanitize_next(next)
    if hmac.compare_digest(username, ADMIN_USERNAME) and hmac.compare_digest(password, ADMIN_PASSWORD):
        response = RedirectResponse(_absolute_next(request, next_raw), status_code=303)
        response.set_cookie(
            key=SESSION_COOKIE_NAME,
            value=_encode_session(),
            max_age=SESSION_MAX_AGE,
            httponly=True,
            secure=os.getenv("ADMIN_COOKIE_SECURE", "true").lower() != "false",
            samesite="lax",
            path="/",
        )
        return response

    context = {"next": next_raw, "error": "Invalid username or password."}
    return _render(request, "admin_login.html", context, status_code=401)


@router.post("/admin/logout", response_class=HTMLResponse, name="admin_logout")
async def admin_logout(
    request: Request,
    next: str | None = Form(default=None, max_length=MAX_TEXT_LENGTH),
):
    next_raw = _sanitize_next(next or request.query_params.get("next"))
    response = RedirectResponse(_absolute_next(request, next_raw), status_code=303)
    response.delete_cookie(SESSION_COOKIE_NAME, path="/")
    return response


@router.post("/tournament/start", response_class=HTMLResponse, name="start_tournament")
async def start_tournament(request: Request, session: Session = Depends(get_session)):
    if not _is_admin(request):
        return _admin_redirect(request)
    event = get_active_event(session)
    teams = session.exec(select(Team).where(Team.event_id == event.id).order_by(Team.created_at)).all()
    if len(teams) < 2:
        redirect_url = str(request.url_for("bracket")) + "?tournament=needs-teams"
        return RedirectResponse(redirect_url, status_code=303)

    _clear_event_matches(session, event.id, keep_playoffs=True)
    session.commit()

    bucket_pool_mode = _needs_bucket_pool(len(teams))
    _ensure_matches(session, event, bucket_pool_mode, teams)
    _refresh_match_statuses(session, event.id)
    redirect_url = str(request.url_for("bracket")) + "?tournament=started"
    return RedirectResponse(redirect_url, status_code=303)


@router.post("/tournament/reset", response_class=HTMLResponse, name="reset_tournament")
async def reset_tournament(request: Request, session: Session = Depends(get_session)):
    if not _is_admin(request):
        return _admin_redirect(request)
    event = get_active_event(session)
    matches = session.exec(select(Match).where(Match.event_id == event.id)).all()
    if not matches:
        redirect_url = str(request.url_for("bracket")) + "?tournament=not-started"
        return RedirectResponse(redirect_url, status_code=303)

    for match in matches:
        match.score_team1 = None
        match.score_team2 = None
        match.status = "pending"
        session.add(match)
    session.commit()

    _refresh_match_statuses(session, event.id)

    redirect_url = str(request.url_for("bracket")) + "?tournament=reset"
    return RedirectResponse(redirect_url, status_code=303)


@router.post("/tournament/start-playoffs", response_class=HTMLResponse, name="start_playoffs")
async def start_playoffs(request: Request, session: Session = Depends(get_session)):
    if not _is_admin(request):
        return _admin_redirect(request)

    event = get_active_event(session)
    snapshot = _schedule_context(session)
    leaderboard = snapshot["leaderboard"]
    group_stage_complete = snapshot.get("group_stage_complete", False)
    semifinal_matches = session.exec(
        select(Match)
        .where(
            (Match.event_id == event.id)
            & (Match.is_playoff == True)
            & (Match.playoff_round == "semifinal")
        )
        .order_by(Match.order_index)
    ).all()
    final_matches = session.exec(
        select(Match)
        .where(
            (Match.event_id == event.id)
            & (Match.is_playoff == True)
            & (Match.playoff_round == "final")
        )
    ).all()

    if not semifinal_matches:
        if not group_stage_complete:
            redirect_url = str(request.url_for("bracket")) + "?playoff=playoff-group-incomplete"
            return RedirectResponse(redirect_url, status_code=303)
        if final_matches:
            for match in final_matches:
                session.delete(match)
            session.commit()

        if len(leaderboard) < 3:
            redirect_url = str(request.url_for("bracket")) + "?playoff=semis-need-teams"
            return RedirectResponse(redirect_url, status_code=303)

        top_four = leaderboard[:4]
        team_ids = [entry["id"] for entry in top_four]
        teams = session.exec(select(Team).where(Team.id.in_(team_ids))).all()
        team_map = {team.id: team for team in teams}
        if len(team_map) < len(top_four):
            redirect_url = str(request.url_for("bracket")) + "?playoff=semis-need-teams"
            return RedirectResponse(redirect_url, status_code=303)

        order_index = 1500
        for entry in top_four:
            team = team_map.get(entry["id"])
            if not team:
                continue
            match = Match(
                event_id=event.id,
                game="Bucket Golf Semifinal",
                order_index=order_index,
                team1_id=team.id,
                team2_id=team.id,
                score_team1=None,
                score_team2=None,
                status="in_progress",
                is_playoff=True,
                playoff_round="semifinal",
            )
            session.add(match)
            order_index += 1
        session.commit()
        _refresh_match_statuses(session, event.id)

        redirect_url = str(request.url_for("bracket")) + "?playoff=semis-created"
        return RedirectResponse(redirect_url, status_code=303)

    incomplete = [match for match in semifinal_matches if match.score_team1 is None]
    if incomplete:
        redirect_url = str(request.url_for("bracket")) + "?playoff=semis-pending"
        return RedirectResponse(redirect_url, status_code=303)

    results = sorted(semifinal_matches, key=lambda match: match.score_team1)
    if len(results) < 2:
        redirect_url = str(request.url_for("bracket")) + "?playoff=semis-pending"
        return RedirectResponse(redirect_url, status_code=303)

    if len(results) > 2 and results[1].score_team1 == results[2].score_team1:
        if final_matches:
            for match in final_matches:
                session.delete(match)
            session.commit()
        redirect_url = str(request.url_for("bracket")) + "?playoff=playoff-tiebreak"
        return RedirectResponse(redirect_url, status_code=303)

    finalist_ids = [results[0].team1_id, results[1].team1_id]
    teams = session.exec(select(Team).where(Team.id.in_(finalist_ids))).all()
    team_map = {team.id: team for team in teams}
    if len(team_map) != 2:
        redirect_url = str(request.url_for("bracket")) + "?playoff=semis-pending"
        return RedirectResponse(redirect_url, status_code=303)

    for match in final_matches:
        session.delete(match)
    session.commit()

    final_order_index = 2000
    final_match = Match(
        event_id=event.id,
        game="Bucket Golf Playoff",
        order_index=final_order_index,
        team1_id=finalist_ids[0],
        team2_id=finalist_ids[1],
        status="in_progress",
        is_playoff=True,
        playoff_round="final",
    )
    session.add(final_match)
    session.commit()

    redirect_url = str(request.url_for("bracket")) + "?playoff=playoff-created"
    return RedirectResponse(redirect_url, status_code=303)


@router.get("/photos", response_class=HTMLResponse, name="photos")
async def photos_page(request: Request, session: Session = Depends(get_session)):
    success = request.query_params.get("photo") == "saved"
    count_param = request.query_params.get("count")
    try:
        success_count = int(count_param) if count_param is not None else None
    except ValueError:
        success_count = None
    if not success:
        success_count = None
    context = _photo_context(session, photo_success=success, photo_success_count=success_count)
    return _render(request, "photos.html", context)


@router.get("/cast", response_class=HTMLResponse, name="cast_display")
async def cast_display(request: Request, session: Session = Depends(get_session)):
    cast_state = _cast_state(session)
    context = {"cast_state": cast_state}
    return _render(request, "cast.html", context)


@router.get("/cast/feed", name="cast_feed")
async def cast_feed(session: Session = Depends(get_session)):
    cast_state = _cast_state(session)
    return JSONResponse(cast_state, headers={"Cache-Control": "no-store"})


@router.get("/events", response_class=HTMLResponse, name="events_page")
async def events_page(request: Request, session: Session = Depends(get_session)):
    cards = _events_context(session)
    context = {"events": cards}
    return _render(request, "events.html", context)


@router.get("/events/{slug}", response_class=HTMLResponse, name="event_detail")
async def event_detail(slug: str, request: Request, session: Session = Depends(get_session)):
    event = session.exec(select(Event).where(Event.slug == slug)).first()
    if not event or event.slug == ACTIVE_EVENT_SLUG:
        raise HTTPException(status_code=404, detail="Event not found")

    photos = session.exec(
        select(Photo)
        .where(Photo.event_id == event.id)
        .order_by(Photo.id.desc())
    ).all()

    winner_photo_url = None
    winner_record = session.exec(
        select(Photo)
        .where((Photo.event_id == event.id) & (func.lower(Photo.original_name).like('winner%')))
        .order_by(Photo.id.desc())
        .limit(1)
    ).first()
    if winner_record:
        winner_photo_url = _photo_image_url(winner_record.filename)
    elif event.winner_photo:
        winner_photo_url = _photo_image_url(event.winner_photo)

    context = {
        "event": event,
        "photos": [_photo_payload(photo) for photo in photos],
        "winner_photo_url": winner_photo_url,
    }
    return _render(request, "event_detail.html", context)


@router.get("/about", response_class=HTMLResponse, name="about_page")
async def about_page(request: Request, session: Session = Depends(get_session)):
    event = get_active_event(session)
    context = {"event": event}
    return _render(request, "about.html", context)


@router.get("/rules", response_class=HTMLResponse, name="game_rules")
async def rules_page(request: Request, session: Session = Depends(get_session)):
    event = get_active_event(session)
    context = {"event": event}
    return _render(request, "game_rules.html", context)


@router.post("/rsvp", response_class=HTMLResponse, name="submit_rsvp")
async def submit_rsvp(
    request: Request,
    session: Session = Depends(get_session),
    name: str = Form(..., max_length=MAX_TEXT_LENGTH),
    guests: int = Form(default=1),
    message: str | None = Form(default=None, max_length=MAX_TEXT_LENGTH),
):
    event = get_active_event(session)
    trimmed_name = name.strip()
    cleaned_message = message.strip() if message else None
    error: str | None = None

    if not trimmed_name:
        error = "Name is required."
    elif len(trimmed_name) > MAX_TEXT_LENGTH:
        error = f"Names must be {MAX_TEXT_LENGTH} characters or fewer."
    elif cleaned_message and len(cleaned_message) > MAX_TEXT_LENGTH:
        error = f"Messages must be {MAX_TEXT_LENGTH} characters or fewer."
    elif guests < 1:
        error = "Please include at least one guest."

    if error:
        context = _home_context(session, rsvp_error=error, include_blocked=_is_admin(request))
        return _render(request, "index.html", context, status_code=400)

    cleaned_message = cleaned_message or None
    flagged = needs_review(trimmed_name, cleaned_message or "")
    status_value = MODERATION_BLOCKED if flagged else MODERATION_APPROVED
    rsvp = RSVP(
        name=trimmed_name,
        email="",
        guests=guests,
        message=cleaned_message,
        event_id=event.id,
        status=status_value,
    )
    session.add(rsvp)
    session.commit()
    notify_admin(
        "RSVP submitted",
        f"Name: {trimmed_name}\nGuests: {guests}\nMessage: {cleaned_message or '-'}\nStatus: {status_value}",
    )

    redirect_state = "pending" if flagged else "saved"
    redirect_url = str(request.url_for("index")) + f"?rsvp={redirect_state}"
    return RedirectResponse(redirect_url, status_code=303)


@router.post("/rsvp/{rsvp_id}/update", response_class=HTMLResponse, name="update_rsvp")
async def update_rsvp(
    request: Request,
    rsvp_id: int,
    session: Session = Depends(get_session),
    name: str = Form(..., max_length=MAX_TEXT_LENGTH),
    guests: int = Form(...),
    message: str | None = Form(default=None, max_length=MAX_TEXT_LENGTH),
):
    rsvp = session.get(RSVP, rsvp_id)
    if not rsvp:
        raise HTTPException(status_code=404, detail="RSVP not found")

    original_name = rsvp.name
    original_guests = rsvp.guests
    original_message = rsvp.message
    original_status = rsvp.status

    trimmed_name = name.strip()
    cleaned_message = message.strip() if message else None
    if not trimmed_name:
        context = _home_context(session, rsvp_error="Name is required.", include_blocked=_is_admin(request))
        return _render(request, "index.html", context, status_code=400)
    if len(trimmed_name) > MAX_TEXT_LENGTH:
        context = _home_context(
            session,
            rsvp_error=f"Names must be {MAX_TEXT_LENGTH} characters or fewer.",
            include_blocked=_is_admin(request),
        )
        return _render(request, "index.html", context, status_code=400)
    if cleaned_message and len(cleaned_message) > MAX_TEXT_LENGTH:
        context = _home_context(
            session,
            rsvp_error=f"Messages must be {MAX_TEXT_LENGTH} characters or fewer.",
            include_blocked=_is_admin(request),
        )
        return _render(request, "index.html", context, status_code=400)
    if guests < 1:
        context = _home_context(session, rsvp_error="Please include at least one guest.", include_blocked=_is_admin(request))
        return _render(request, "index.html", context, status_code=400)

    rsvp.name = trimmed_name
    rsvp.guests = guests
    updated_message = cleaned_message or None
    rsvp.message = updated_message

    flagged_update = needs_review(trimmed_name, updated_message or "")
    if flagged_update:
        upsert_pending_change(
            session,
            "rsvp",
            rsvp.id,
            {
                "name": original_name,
                "guests": original_guests,
                "message": original_message,
                "status": original_status,
            },
            {
                "name": trimmed_name,
                "guests": guests,
                "message": updated_message,
                "status": MODERATION_APPROVED,
            },
        )
        rsvp.name = original_name
        rsvp.guests = original_guests
        rsvp.message = original_message
        rsvp.status = original_status
        redirect_state = "pending"
    else:
        rsvp.status = MODERATION_APPROVED
        delete_pending_change(session, "rsvp", rsvp.id)
        redirect_state = "updated"

    session.add(rsvp)
    session.commit()

    if flagged_update:
        notify_admin(
            "RSVP update pending review",
            (
                "Original entry:\\n"
                f"  Name: {original_name}\\n"
                f"  Guests: {original_guests}\\n"
                f"  Message: {(original_message or '-')}"
                "\\n\\nRequested changes:\\n"
                f"  Name: {trimmed_name}\\n"
                f"  Guests: {guests}\\n"
                f"  Message: {(updated_message or '-')}\\n"
                f"Status: {rsvp.status}"
            ),
        )
    else:
        notify_admin(
            "RSVP updated",
            f"Name: {trimmed_name}\\nGuests: {guests}\\nMessage: {(updated_message or '-')}\\nStatus: {rsvp.status}",
        )

    redirect_url = str(request.url_for("index")) + f"?rsvp={redirect_state}"
    return RedirectResponse(redirect_url, status_code=303)


@router.post("/rsvp/{rsvp_id}/delete", response_class=HTMLResponse, name="delete_rsvp")
async def delete_rsvp(request: Request, rsvp_id: int, session: Session = Depends(get_session)):
    rsvp = session.get(RSVP, rsvp_id)
    if not rsvp:
        raise HTTPException(status_code=404, detail="RSVP not found")
    delete_pending_change(session, "rsvp", rsvp_id)
    session.delete(rsvp)
    session.commit()
    notify_admin("RSVP deleted", f"Name: {rsvp.name}\\nGuests: {rsvp.guests}")

    redirect_url = str(request.url_for("index")) + "?rsvp=deleted"
    return RedirectResponse(redirect_url, status_code=303)


@router.post("/admin/rsvp/{rsvp_id}/approve", response_class=HTMLResponse, name="admin_approve_rsvp")
async def admin_approve_rsvp(
    request: Request,
    rsvp_id: int,
    session: Session = Depends(get_session),
    next: str | None = Form(default=None, max_length=MAX_TEXT_LENGTH),
):
    if not _is_admin(request):
        return _admin_redirect(request)
    rsvp = session.get(RSVP, rsvp_id)
    if not rsvp:
        raise HTTPException(status_code=404, detail="RSVP not found")
    change = get_pending_change(session, "rsvp", rsvp_id)
    if change:
        proposed = json.loads(change.proposed_data)
        rsvp.name = proposed.get("name", rsvp.name)
        rsvp.guests = proposed.get("guests", rsvp.guests)
        rsvp.message = proposed.get("message", rsvp.message)
        delete_pending_change(session, "rsvp", rsvp_id)
        note = "Approved pending update"
    else:
        note = "Approved submission"
    rsvp.status = MODERATION_APPROVED
    session.add(rsvp)
    session.commit()
    notify_admin(
        "RSVP approved",
        f"{note}:\\nName: {rsvp.name}\\nGuests: {rsvp.guests}\\nMessage: {rsvp.message or '-'}",
    )
    target = next or request.url_for("index")
    return RedirectResponse(str(target), status_code=303)


@router.post("/admin/rsvp/{rsvp_id}/deny", response_class=HTMLResponse, name="admin_deny_rsvp")
async def admin_deny_rsvp(
    request: Request,
    rsvp_id: int,
    session: Session = Depends(get_session),
    next: str | None = Form(default=None, max_length=MAX_TEXT_LENGTH),
):
    if not _is_admin(request):
        return _admin_redirect(request)
    rsvp = session.get(RSVP, rsvp_id)
    if not rsvp:
        raise HTTPException(status_code=404, detail="RSVP not found")
    change = get_pending_change(session, "rsvp", rsvp_id)
    if change:
        original = json.loads(change.original_data)
        proposed = json.loads(change.proposed_data)
        rsvp.name = original.get("name", rsvp.name)
        rsvp.guests = original.get("guests", rsvp.guests)
        rsvp.message = original.get("message", rsvp.message)
        rsvp.status = original.get("status", MODERATION_APPROVED)
        delete_pending_change(session, "rsvp", rsvp_id)
        session.add(rsvp)
        session.commit()
        notify_admin(
            "RSVP update denied",
            (
                "Reverted RSVP to original details.\n"
                f"Original: {original.get('name', '-')}, guests {original.get('guests', '-')}, message {original.get('message') or '-'}\n"
                f"Rejected changes: {proposed.get('name', '-')}, guests {proposed.get('guests', '-')}, message {proposed.get('message') or '-'}"
            ),
        )
    else:
        session.delete(rsvp)
        session.commit()
        notify_admin("RSVP denied", f"Removed pending RSVP: {rsvp.name}")
    target = next or request.url_for("index")
    return RedirectResponse(str(target), status_code=303)


@router.post("/photos", response_class=HTMLResponse, name="upload_photo")
async def upload_photo(
    request: Request,
    session: Session = Depends(get_session),
    images: list[UploadFile] = File(...),
):
    event = get_active_event(session)

    if not images:
        context = _photo_context(session, photo_error="Select at least one image to upload.")
        return _render(request, "photos.html", context, status_code=400)

    allowed_suffixes = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".heic", ".heif"}
    prepared: list[dict[str, object]] = []
    errors: list[str] = []

    for image in images:
        original_name = image.filename or "upload.png"
        content_type = (image.content_type or "").lower()
        suffix = Path(original_name).suffix.lower() or ".png"

        if not content_type.startswith("image/"):
            errors.append(f"{original_name}: only image uploads are allowed.")
            continue
        if suffix not in allowed_suffixes:
            errors.append(f"{original_name}: use PNG, JPG, GIF, HEIC, or WebP images.")
            continue

        try:
            file_bytes = await image.read()
        except Exception:
            errors.append(f"{original_name}: failed to read upload.")
            continue

        if suffix in {".heic", ".heif"}:
            if not read_heif or not Image:
                errors.append(f"{original_name}: HEIC support is not available on the server.")
                continue
            try:
                heif_file = read_heif(file_bytes)
                img = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data, "raw")
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                file_bytes = buffer.getvalue()
                suffix = ".jpg"
                content_type = "image/jpeg"
                original_name = f"{Path(original_name).stem}.jpg"
            except Exception:
                errors.append(f"{original_name}: could not convert HEIC image.")
                continue
        else:
            if suffix in {".jpg", ".jpeg"}:
                content_type = "image/jpeg"

        prepared.append(
            {
                "data": file_bytes,
                "original_name": original_name,
                "suffix": suffix,
                "content_type": content_type or "image/jpeg",
            }
        )

    if not prepared:
        context = _photo_context(session, photo_error=" ".join(errors))
        return _render(request, "photos.html", context, status_code=400)

    successes = 0
    upload_failures: list[str] = []

    for item in prepared:
        data = item["data"]
        original_name = item["original_name"]
        suffix = item["suffix"]
        content_type = item["content_type"]
        object_basename = f"{uuid4().hex}{suffix}"
        storage_identifier = object_basename
        try:
            if USE_GCS_PHOTOS:
                object_name = f"{event.slug}/photos/{object_basename}"
                upload_photo_stream(BytesIO(data), object_name=object_name, content_type=content_type)
                storage_identifier = make_gcs_identifier(object_name)
            else:
                _ensure_upload_dir()
                destination = UPLOAD_DIR / object_basename
                with destination.open("wb") as buffer:
                    buffer.write(data)

            photo = Photo(filename=storage_identifier, original_name=original_name, event_id=event.id)
            session.add(photo)
            session.commit()
            successes += 1
        except RuntimeError as exc:
            logger.exception("Photo upload failed: %s", exc)
            upload_failures.append(f"{original_name}: {exc}")
            session.rollback()
        except Exception:
            logger.exception("Unexpected error while uploading photo %s.", original_name)
            upload_failures.append(
                f"{original_name}: we hit a snag saving that photo. Please try again."
            )
            session.rollback()

    if upload_failures:
        error_msg = " ".join(upload_failures)
        context = _photo_context(
            session,
            photo_error=error_msg,
            photo_success=successes > 0,
            photo_success_count=successes if successes else None,
        )
        return _render(request, "photos.html", context, status_code=207 if successes else 500)

    redirect_url = str(request.url_for("photos")) + f"?photo=saved&count={successes}"
    return RedirectResponse(redirect_url, status_code=303)


@router.get("/teams", response_class=HTMLResponse, name="team_directory")
async def team_directory(request: Request, session: Session = Depends(get_session)):
    success_message = None
    form_error = None

    context = _team_context(
        session=session,
        form_error=form_error,
        form_value="",
        form_member_one="",
        form_member_two="",
        success_message=success_message,
        is_admin=_is_admin(request),
    )

    team_status = request.query_params.get("team")
    if team_status == "created":
        context["success_message"] = "Team added successfully."
    elif team_status == "updated":
        context["success_message"] = "Team updated."
    elif team_status == "deleted":
        context["success_message"] = "Team removed."
    elif team_status == "exists":
        context["form_error"] = "That team name already exists."
    elif team_status == "invalid":
        context["form_error"] = "Team names must be at least two characters long."
    elif team_status == "pending":
        context["success_message"] = "Thanks! Your team will appear once an organizer approves it."
    elif team_status == "too-long":
        context["form_error"] = f"Team and player names must be {MAX_TEXT_LENGTH} characters or fewer."

    flag = request.query_params.get("free_agent")
    if flag == "added":
        context["free_agent_success"] = "Added to the free-agent pool. We'll match you up soon."
    elif flag == "paired":
        context["free_agent_success"] = "Matched free agents and created a new team!"
    elif flag == "updated":
        context["free_agent_success"] = "Free agent updated."
    elif flag == "deleted":
        context["free_agent_success"] = "Free agent removed."
    elif flag == "error":
        context["free_agent_error"] = "Unable to update the free agent. Provide a name."
    elif flag == "pending":
        context["free_agent_success"] = "Thanks! We'll review that free-agent request shortly."
    elif flag == "too-long":
        context["free_agent_error"] = f"Free-agent details must be {MAX_TEXT_LENGTH} characters or fewer."

    context.setdefault("free_agent_error", None)
    context.setdefault("free_agent_success", None)
    return _render(request, "teams.html", context)


@router.post("/teams", response_class=HTMLResponse, name="create_team")
async def create_team(
    request: Request,
    session: Session = Depends(get_session),
    name: str = Form(default="", max_length=MAX_TEXT_LENGTH),
    member_one: str = Form(default="", max_length=MAX_TEXT_LENGTH),
    member_two: str = Form(default="", max_length=MAX_TEXT_LENGTH),
):
    cleaned = name.strip()
    first_player = member_one.strip() or None
    second_player = member_two.strip() or None
    error: str | None = None
    event = get_active_event(session)

    if len(cleaned) < 2:
        error = "Team names must be at least two characters long."
    elif len(cleaned) > MAX_TEXT_LENGTH:
        error = f"Team names must be {MAX_TEXT_LENGTH} characters or fewer."
    elif first_player and len(first_player) > MAX_TEXT_LENGTH:
        error = f"Player names must be {MAX_TEXT_LENGTH} characters or fewer."
    elif second_player and len(second_player) > MAX_TEXT_LENGTH:
        error = f"Player names must be {MAX_TEXT_LENGTH} characters or fewer."
    else:
        existing = session.exec(select(Team).where((Team.name == cleaned) & (Team.event_id == event.id))).first()
        if existing:
            error = "That team already exists."

    if error:
        context = _team_context(
            session=session,
            form_error=error,
            form_value=name,
            form_member_one=member_one,
            form_member_two=member_two,
            success_message=None,
            is_admin=_is_admin(request),
        )
        return _render(request, "teams.html", context)

    flagged = needs_review(cleaned, first_player or "", second_player or "")
    team_status = MODERATION_BLOCKED if flagged else MODERATION_APPROVED
    team = Team(
        name=cleaned,
        event_id=event.id,
        member_one=first_player,
        member_two=second_player,
        status=team_status,
    )
    session.add(team)
    session.commit()
    session.refresh(team)

    if not flagged:
        _clear_event_matches(session, event.id)
        session.commit()
    notify_admin(
        "Team created",
        f"Team: {cleaned}\nMembers: {(first_player or 'TBD')} & {(second_player or 'TBD')}\nStatus: {team_status}",
    )

    redirect_state = "pending" if flagged else "created"
    redirect_url = str(request.url_for("team_directory")) + f"?team={redirect_state}"
    return RedirectResponse(redirect_url, status_code=303)


@router.post("/teams/{team_id}/update", response_class=HTMLResponse, name="update_team")
async def update_team(
    request: Request,
    team_id: int,
    session: Session = Depends(get_session),
    name: str = Form(..., max_length=MAX_TEXT_LENGTH),
    member_one: str = Form(..., max_length=MAX_TEXT_LENGTH),
    member_two: str = Form(..., max_length=MAX_TEXT_LENGTH),
):
    team = session.get(Team, team_id)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")

    original_name = team.name
    original_member_one = team.member_one
    original_member_two = team.member_two
    original_status = team.status

    cleaned = name.strip()
    first_player = member_one.strip() or None
    second_player = member_two.strip() or None
    if len(cleaned) < 2:
        redirect_url = str(request.url_for("team_directory")) + "?team=invalid"
        return RedirectResponse(redirect_url, status_code=303)
    if len(cleaned) > MAX_TEXT_LENGTH or (
        first_player and len(first_player) > MAX_TEXT_LENGTH
    ) or (second_player and len(second_player) > MAX_TEXT_LENGTH):
        redirect_url = str(request.url_for("team_directory")) + "?team=too-long"
        return RedirectResponse(redirect_url, status_code=303)

    duplicate = session.exec(
        select(Team).where((Team.id != team.id) & (Team.event_id == team.event_id) & (Team.name == cleaned))
    ).first()
    if duplicate:
        redirect_url = str(request.url_for("team_directory")) + "?team=exists"
        return RedirectResponse(redirect_url, status_code=303)

    flagged = needs_review(cleaned, first_player or "", second_player or "")
    proposed_payload = {
        "name": cleaned,
        "member_one": first_player,
        "member_two": second_player,
        "status": MODERATION_APPROVED,
    }
    if flagged:
        upsert_pending_change(
            session,
            "team",
            team.id,
            {
                "name": original_name,
                "member_one": original_member_one,
                "member_two": original_member_two,
                "status": original_status,
            },
            proposed_payload,
        )
        team.name = original_name
        team.member_one = original_member_one
        team.member_two = original_member_two
        team.status = original_status
        session.add(team)
        session.commit()
        redirect_state = "pending"
    else:
        team.name = proposed_payload["name"]
        team.member_one = proposed_payload["member_one"]
        team.member_two = proposed_payload["member_two"]
        team.status = MODERATION_APPROVED
        delete_pending_change(session, "team", team.id)
        session.add(team)
        session.commit()
        _clear_event_matches(session, team.event_id)
        session.commit()
        redirect_state = "updated"

    if flagged:
        notify_admin(
            "Team update pending review",
            (
                "Original entry:\n"
                f"  Team: {original_name}\n"
                f"  Members: {(original_member_one or 'TBD')} & {(original_member_two or 'TBD')}\n"
                f"  Status: {original_status}\n\n"
                "Requested changes:\n"
                f"  Team: {proposed_payload['name']}\n"
                f"  Members: {(proposed_payload['member_one'] or 'TBD')} & {(proposed_payload['member_two'] or 'TBD')}\n"
                f"  Status: {proposed_payload['status']}"
            ),
        )
    else:
        notify_admin(
            "Team updated",
            f"Team: {team.name}\nMembers: {(team.member_one or 'TBD')} & {(team.member_two or 'TBD')}\nStatus: {team.status}",
        )

    redirect_url = str(request.url_for("team_directory")) + f"?team={redirect_state}"
    return RedirectResponse(redirect_url, status_code=303)


@router.post("/teams/{team_id}/delete", response_class=HTMLResponse, name="delete_team")
async def delete_team(request: Request, team_id: int, session: Session = Depends(get_session)):
    team = session.get(Team, team_id)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")

    event_id = team.event_id
    agents = session.exec(select(FreeAgent).where(FreeAgent.team_id == team.id)).all()
    for agent in agents:
        agent.team_id = None
        agent.status = "pending"
        session.add(agent)

    delete_pending_change(session, "team", team_id)
    _clear_event_matches(session, event_id)
    session.delete(team)
    session.commit()
    notify_admin("Team deleted", f"Team: {team.name}")

    redirect_url = str(request.url_for("team_directory")) + "?team=deleted"
    return RedirectResponse(redirect_url, status_code=303)


@router.post("/admin/team/{team_id}/approve", response_class=HTMLResponse, name="admin_approve_team")
async def admin_approve_team(
    request: Request,
    team_id: int,
    session: Session = Depends(get_session),
    next: str | None = Form(default=None, max_length=MAX_TEXT_LENGTH),
):
    if not _is_admin(request):
        return _admin_redirect(request)
    team = session.get(Team, team_id)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")
    change = get_pending_change(session, "team", team_id)
    if change:
        proposed = json.loads(change.proposed_data)
        team.name = proposed.get("name", team.name)
        team.member_one = proposed.get("member_one", team.member_one)
        team.member_two = proposed.get("member_two", team.member_two)
        delete_pending_change(session, "team", team_id)
        note = "Approved pending update"
    else:
        note = "Approved submission"
    team.status = MODERATION_APPROVED
    session.add(team)
    session.commit()
    _clear_event_matches(session, team.event_id)
    session.commit()
    notify_admin("Team approved", f"{note}: {team.name}")
    target = next or request.url_for("team_directory")
    return RedirectResponse(str(target), status_code=303)


@router.post("/admin/team/{team_id}/deny", response_class=HTMLResponse, name="admin_deny_team")
async def admin_deny_team(
    request: Request,
    team_id: int,
    session: Session = Depends(get_session),
    next: str | None = Form(default=None, max_length=MAX_TEXT_LENGTH),
):
    if not _is_admin(request):
        return _admin_redirect(request)
    team = session.get(Team, team_id)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")
    change = get_pending_change(session, "team", team_id)
    if change:
        original = json.loads(change.original_data)
        proposed = json.loads(change.proposed_data)
        team.name = original.get("name", team.name)
        team.member_one = original.get("member_one", team.member_one)
        team.member_two = original.get("member_two", team.member_two)
        team.status = original.get("status", MODERATION_APPROVED)
        delete_pending_change(session, "team", team_id)
        session.add(team)
        session.commit()
        notify_admin(
            "Team update denied",
            (
                "Reverted team to original details.\n"
                f"Original name: {original.get('name', '-')}\n"
                f"Original members: {(original.get('member_one') or 'TBD')} & {(original.get('member_two') or 'TBD')}\n"
                f"Rejected name: {proposed.get('name', '-')}\n"
                f"Rejected members: {(proposed.get('member_one') or 'TBD')} & {(proposed.get('member_two') or 'TBD')}"
            ),
        )
    else:
        session.delete(team)
        session.commit()
        notify_admin("Team denied", f"Removed pending team: {team.name}")
    target = next or request.url_for("team_directory")
    return RedirectResponse(str(target), status_code=303)


@router.post("/free-agent", response_class=HTMLResponse, name="register_free_agent")
async def register_free_agent(
    request: Request,
    session: Session = Depends(get_session),
    name: str = Form(..., max_length=MAX_TEXT_LENGTH),
    email: str | None = Form(default=None, max_length=MAX_TEXT_LENGTH),
    note: str | None = Form(default=None, max_length=MAX_TEXT_LENGTH),
):
    trimmed_name = name.strip()
    trimmed_email = email.strip() if email else None
    cleaned_note = note.strip() if note else None
    event = get_active_event(session)

    if not trimmed_name:
        context = _team_context(
            session=session,
            form_error=None,
            form_value="",
            form_member_one="",
            form_member_two="",
            success_message=None,
            is_admin=_is_admin(request),
        )
        context["free_agent_error"] = "Name is required for free agents."
        context.setdefault("free_agent_success", None)
        return _render(request, "teams.html", context, status_code=400)
    if len(trimmed_name) > MAX_TEXT_LENGTH:
        context = _team_context(
            session=session,
            form_error=None,
            form_value="",
            form_member_one="",
            form_member_two="",
            success_message=None,
            is_admin=_is_admin(request),
        )
        context["free_agent_error"] = f"Free-agent names must be {MAX_TEXT_LENGTH} characters or fewer."
        context.setdefault("free_agent_success", None)
        return _render(request, "teams.html", context, status_code=400)
    if trimmed_email and len(trimmed_email) > MAX_TEXT_LENGTH:
        context = _team_context(
            session=session,
            form_error=None,
            form_value="",
            form_member_one="",
            form_member_two="",
            success_message=None,
            is_admin=_is_admin(request),
        )
        context["free_agent_error"] = f"Emails must be {MAX_TEXT_LENGTH} characters or fewer."
        context.setdefault("free_agent_success", None)
        return _render(request, "teams.html", context, status_code=400)
    if cleaned_note and len(cleaned_note) > MAX_TEXT_LENGTH:
        context = _team_context(
            session=session,
            form_error=None,
            form_value="",
            form_member_one="",
            form_member_two="",
            success_message=None,
            is_admin=_is_admin(request),
        )
        context["free_agent_error"] = f"Notes must be {MAX_TEXT_LENGTH} characters or fewer."
        context.setdefault("free_agent_success", None)
        return _render(request, "teams.html", context, status_code=400)

    cleaned_email = trimmed_email or ""
    flagged = needs_review(trimmed_name, cleaned_email, cleaned_note or "")
    moderation_status = MODERATION_BLOCKED if flagged else MODERATION_APPROVED
    agent = FreeAgent(
        name=trimmed_name,
        email=cleaned_email,
        note=cleaned_note,
        event_id=event.id,
        moderation_status=moderation_status,
    )
    session.add(agent)
    session.commit()
    notify_admin(
        "Free agent joined",
        f"Name: {trimmed_name}\nEmail: {trimmed_email or '-'}\nNote: {agent.note or '-'}\nStatus: {moderation_status}",
    )

    redirect_state = "pending" if flagged else "added"
    redirect_url = str(request.url_for("team_directory")) + f"?free_agent={redirect_state}"
    return RedirectResponse(redirect_url, status_code=303)


@router.post("/free-agent/{agent_id}/update", response_class=HTMLResponse, name="update_free_agent")
async def update_free_agent(
    request: Request,
    agent_id: int,
    session: Session = Depends(get_session),
    name: str = Form(..., max_length=MAX_TEXT_LENGTH),
    status: str = Form(..., max_length=MAX_TEXT_LENGTH),
    note: str | None = Form(default=None, max_length=MAX_TEXT_LENGTH),
    team_id: str | None = Form(default=None, max_length=MAX_TEXT_LENGTH),
    pair_with: str | None = Form(default=None, max_length=MAX_TEXT_LENGTH),
):
    agent = session.get(FreeAgent, agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Free agent not found")

    original_name = agent.name
    original_note = agent.note
    original_status = agent.status
    original_team_id = agent.team_id
    original_moderation = agent.moderation_status

    trimmed_name = name.strip()
    cleaned_note = note.strip() if note else None
    if not trimmed_name:
        redirect_url = str(request.url_for("team_directory")) + "?free_agent=error"
        return RedirectResponse(redirect_url, status_code=303)
    if len(trimmed_name) > MAX_TEXT_LENGTH or (cleaned_note and len(cleaned_note) > MAX_TEXT_LENGTH):
        redirect_url = str(request.url_for("team_directory")) + "?free_agent=too-long"
        return RedirectResponse(redirect_url, status_code=303)

    status_value = status.lower()
    if status_value not in ALLOWED_FREE_AGENT_STATUSES:
        status_value = "pending"
    requested_status = status_value

    requested_team_id: int | None = None
    requested_team_name: str | None = None
    if team_id:
        try:
            candidate_id = int(team_id)
        except ValueError:
            candidate_id = None
        if candidate_id is not None:
            candidate_team = session.get(Team, candidate_id)
            if candidate_team and candidate_team.event_id == agent.event_id:
                requested_team_id = candidate_team.id
                requested_team_name = candidate_team.name

    flagged = needs_review(trimmed_name, agent.email or "", cleaned_note or "")
    moderation_status = MODERATION_BLOCKED if flagged else MODERATION_APPROVED

    proposed_payload = {
        "name": trimmed_name,
        "note": cleaned_note,
        "status": requested_status,
        "team_id": requested_team_id,
        "moderation_status": MODERATION_APPROVED,
    }

    if flagged:
        upsert_pending_change(
            session,
            "freeagent",
            agent.id,
            {
                "name": original_name,
                "note": original_note,
                "status": original_status,
                "team_id": original_team_id,
                "moderation_status": original_moderation,
            },
            proposed_payload,
        )
        agent.name = original_name
        agent.note = original_note
        agent.status = original_status
        agent.team_id = original_team_id
        agent.moderation_status = original_moderation
        session.add(agent)
        session.commit()
        original_team_label = _team_display_name(session, original_team_id)
        requested_team_label = requested_team_name or "-- No team --"
        notify_admin(
            "Free agent update pending",
            (
                "Original entry:\n"
                f"  Name: {original_name}\n"
                f"  Email: {agent.email or '-'}\n"
                f"  Status: {original_status}\n"
                f"  Note: {(original_note or '-')}\n"
                f"  Team: {original_team_label}\n"
                f"  Moderation: {original_moderation}\n\n"
                "Requested changes:\n"
                f"  Name: {proposed_payload['name']}\n"
                f"  Email: {agent.email or '-'}\n"
                f"  Status: {proposed_payload['status']}\n"
                f"  Note: {(proposed_payload['note'] or '-')}\n"
                f"  Team: {requested_team_label}\n"
                f"  Moderation: {proposed_payload['moderation_status']}"
            ),
        )
        redirect_url = str(request.url_for("team_directory")) + "?free_agent=pending"
        return RedirectResponse(redirect_url, status_code=303)
    else:
        agent.name = proposed_payload["name"]
        agent.note = proposed_payload["note"]
        agent.status = requested_status
        agent.team_id = requested_team_id
        agent.moderation_status = moderation_status

    # Optional: pair with another free agent to create a new team
    if pair_with:
        try:
            partner_id = int(pair_with)
        except ValueError:
            partner_id = None
        if partner_id and partner_id != agent.id:
            partner = session.get(FreeAgent, partner_id)
        else:
            partner = None
        if partner and partner.event_id == agent.event_id and partner.status == "pending":
            team_name = _generate_free_agent_team_name(session, agent, partner)
            team = Team(
                name=team_name,
                event_id=agent.event_id,
                member_one=agent.name.strip() or agent.name,
                member_two=partner.name.strip() or partner.name,
            )
            session.add(team)
            session.commit()
            session.refresh(team)

            agent.status = "paired"
            agent.team_id = team.id
            partner.status = "paired"
            partner.team_id = team.id
            session.add(partner)
            session.add(agent)
            session.commit()

            _clear_event_matches(session, agent.event_id)
            session.commit()

            redirect_url = str(request.url_for("team_directory")) + "?free_agent=paired"
        return RedirectResponse(redirect_url, status_code=303)

    if requested_team_id is None or requested_status == "pending":
        agent.team_id = None
        agent.status = "pending"
    else:
        agent.team_id = requested_team_id
        agent.status = "paired"

    session.add(agent)
    delete_pending_change(session, "freeagent", agent.id)
    session.commit()

    notify_admin(
        "Free agent updated",
        f"Name: {agent.name}\nEmail: {agent.email or '-'}\nNote: {agent.note or '-'}\nStatus: {agent.status}\nModeration: {agent.moderation_status}",
    )

    redirect_url = str(request.url_for("team_directory")) + "?free_agent=updated"
    return RedirectResponse(redirect_url, status_code=303)


@router.post("/free-agent/{agent_id}/delete", response_class=HTMLResponse, name="delete_free_agent")
async def delete_free_agent(request: Request, agent_id: int, session: Session = Depends(get_session)):
    agent = session.get(FreeAgent, agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Free agent not found")
    delete_pending_change(session, "freeagent", agent_id)
    session.delete(agent)
    session.commit()
    notify_admin("Free agent deleted", f"Name: {agent.name}\nEmail: {agent.email or '-'}")

    redirect_url = str(request.url_for("team_directory")) + "?free_agent=deleted"
    return RedirectResponse(redirect_url, status_code=303)


@router.post(
    "/admin/free-agent/{agent_id}/approve",
    response_class=HTMLResponse,
    name="admin_approve_free_agent",
)
async def admin_approve_free_agent(
    request: Request,
    agent_id: int,
    session: Session = Depends(get_session),
    next: str | None = Form(default=None, max_length=MAX_TEXT_LENGTH),
):
    if not _is_admin(request):
        return _admin_redirect(request)
    agent = session.get(FreeAgent, agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Free agent not found")
    change = get_pending_change(session, "freeagent", agent_id)
    if change:
        proposed = json.loads(change.proposed_data)
        agent.name = proposed.get("name", agent.name)
        agent.note = proposed.get("note", agent.note)
        agent.status = proposed.get("status", agent.status)
        agent.team_id = proposed.get("team_id", agent.team_id)
        delete_pending_change(session, "freeagent", agent_id)
        note = "Approved pending update"
    else:
        note = "Approved submission"
    agent.moderation_status = MODERATION_APPROVED
    session.add(agent)
    session.commit()
    notify_admin(
        "Free agent approved",
        f"{note}: {agent.name}\nEmail: {agent.email or '-'}\nStatus: {agent.status}",
    )
    target = next or request.url_for("team_directory")
    return RedirectResponse(str(target), status_code=303)


@router.post(
    "/admin/free-agent/{agent_id}/deny",
    response_class=HTMLResponse,
    name="admin_deny_free_agent",
)
async def admin_deny_free_agent(
    request: Request,
    agent_id: int,
    session: Session = Depends(get_session),
    next: str | None = Form(default=None, max_length=MAX_TEXT_LENGTH),
):
    if not _is_admin(request):
        return _admin_redirect(request)
    agent = session.get(FreeAgent, agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Free agent not found")
    change = get_pending_change(session, "freeagent", agent_id)
    if change:
        original = json.loads(change.original_data)
        proposed = json.loads(change.proposed_data)
        agent.name = original.get("name", agent.name)
        agent.note = original.get("note", agent.note)
        agent.status = original.get("status", agent.status)
        agent.team_id = original.get("team_id", agent.team_id)
        agent.moderation_status = original.get("moderation_status", MODERATION_APPROVED)
        delete_pending_change(session, "freeagent", agent_id)
        session.add(agent)
        session.commit()
        notify_admin(
            "Free agent update denied",
            (
                "Reverted free agent to original details.\n"
                f"Original name: {original.get('name', '-')}\n"
                f"Original status: {original.get('status', '-')}\n"
                f"Original team: {_team_display_name(session, original.get('team_id'))}\n"
                f"Rejected name: {proposed.get('name', '-')}\n"
                f"Rejected status: {proposed.get('status', '-')}\n"
                f"Rejected team: {_team_display_name(session, proposed.get('team_id'))}"
            ),
        )
    else:
        session.delete(agent)
        session.commit()
        notify_admin("Free agent denied", f"Removed pending free agent: {agent.name}")
    target = next or request.url_for("team_directory")
    return RedirectResponse(str(target), status_code=303)


@router.post("/matches/{match_id}/score", response_class=HTMLResponse, name="record_score")
async def record_score(
    request: Request,
    match_id: int,
    session: Session = Depends(get_session),
    score_team1: int = Form(...),
    score_team2: int = Form(...),
):
    match = session.exec(select(Match).where(Match.id == match_id)).first()
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")

    is_admin = _is_admin(request)
    completed_already = match.score_team1 is not None and match.score_team2 is not None
    if completed_already and not is_admin:
        redirect_url = str(request.url_for("bracket")) + "?tournament=score-locked"
        return RedirectResponse(redirect_url, status_code=303)

    if score_team1 < 0 or score_team2 < 0:
        raise HTTPException(status_code=400, detail="Scores must be zero or greater")

    match.score_team1 = score_team1
    match.score_team2 = score_team2
    match.status = "completed"
    session.add(match)
    session.commit()

    _refresh_match_statuses(session, match.event_id)
    redirect_url = str(request.url_for("bracket")) + "?score=updated"
    return RedirectResponse(redirect_url, status_code=303)


def _home_context(
    session: Session,
    *,
    rsvp_error: str | None = None,
    rsvp_status: str | None = None,
    include_blocked: bool = False,
) -> dict[str, object]:
    event = get_active_event(session)
    rsvps = _fetch_rsvps(session, event.id, status=MODERATION_APPROVED)
    guest_total = sum(r.guests for r in rsvps)
    return {
        "event": event,
        "rsvps": rsvps,
        "guest_total": guest_total,
        "rsvp_error": rsvp_error,
        "rsvp_status": rsvp_status,
        "blocked_rsvps": _fetch_rsvps(session, event.id, status=MODERATION_BLOCKED) if include_blocked else [],
        "pending_rsvp_updates": _pending_updates(session, "rsvp", RSVP) if include_blocked else [],
    }


def _photo_context(
    session: Session,
    *,
    photo_error: str | None = None,
    photo_success: bool = False,
    photo_success_count: int | None = None,
) -> dict[str, object]:
    event = get_active_event(session)
    photos = [_photo_payload(photo) for photo in _fetch_photos(session, event.id)]
    return {
        "event": event,
        "photos": photos,
        "photo_error": photo_error,
        "photo_success": photo_success,
        "photo_success_count": photo_success_count,
    }


def _events_context(session: Session) -> list[dict[str, object]]:
    events = session.exec(
        select(Event)
        .where(Event.slug != ACTIVE_EVENT_SLUG)
        .order_by(Event.event_date.desc())
    ).all()
    cards: list[dict[str, object]] = []
    for event in events:
        team_count = len(
            session.exec(
                select(Team)
                .where((Team.event_id == event.id) & (Team.status == MODERATION_APPROVED))
            ).all()
        )
        photo_query = (
            select(Photo)
            .where(Photo.event_id == event.id)
            .order_by(Photo.created_at.desc())
        )
        preview_results = session.exec(photo_query.limit(EVENT_GALLERY_PREVIEW_LIMIT + 1)).all()
        has_more = len(preview_results) > EVENT_GALLERY_PREVIEW_LIMIT
        preview_photos = [
            _photo_payload(photo)
            for photo in sorted(preview_results[:EVENT_GALLERY_PREVIEW_LIMIT], key=lambda p: p.id, reverse=True)
        ]
        winner_image_url = None
        winner_record = session.exec(
            select(Photo)
            .where((Photo.event_id == event.id) & (func.lower(Photo.original_name).like('winner%')))
            .order_by(Photo.id.desc())
            .limit(1)
        ).first()
        if winner_record:
            winner_image_url = _photo_image_url(winner_record.filename)
        elif event.winner_photo:
            winner_image_url = _photo_image_url(event.winner_photo)
        cards.append(
            {
                "event": event,
                "photos": preview_photos,
                "team_count": team_count,
                "has_more": has_more,
                "detail_slug": event.slug,
                "winner_image_url": winner_image_url,
            }
        )
    return cards


def _needs_bucket_pool(team_count: int) -> bool:
    if team_count <= 0:
        return False
    return team_count < len(GAMES) + 1 or team_count % 2 == 1


def _schedule_context(session: Session) -> dict[str, object]:
    event = get_active_event(session)
    teams = session.exec(
        select(Team)
        .where((Team.event_id == event.id) & (Team.status == MODERATION_APPROVED))
        .order_by(Team.created_at)
    ).all()
    bucket_pool_mode = _needs_bucket_pool(len(teams))
    matches = _ensure_matches(session, event, bucket_pool_mode, teams)
    _refresh_match_statuses(session, event.id)
    matches = session.exec(
        select(Match)
        .where((Match.event_id == event.id) & (Match.is_playoff == False))
        .order_by(Match.order_index)
    ).all()
    team_lookup = {team.id: team.name for team in teams}

    match_payload = [
        {
            "id": match.id,
            "game": match.game,
            "order": match.order_index,
            "status": match.status,
            "team1": team_lookup.get(match.team1_id),
            "team2": team_lookup.get(match.team2_id),
            "team1_id": match.team1_id,
            "team2_id": match.team2_id,
            "score1": match.score_team1,
            "score2": match.score_team2,
        }
        for match in matches
    ]

    matches_by_game = []
    next_matches = []
    game_participants: dict[str, set[int]] = {game: set() for game in GAMES}
    for game in GAMES:
        grouped = [payload for payload in match_payload if payload["game"] == game]
        if not grouped:
            continue
        # Determine next match for this game.
        next_match = next((payload for payload in grouped if payload["status"] != "completed"), None)
        if next_match:
            next_matches.append({"game": game, "match": next_match})
        for payload in grouped:
            game_participants.setdefault(game, set()).add(payload["team1_id"])
            if payload["team2_id"] != payload["team1_id"]:
                game_participants.setdefault(game, set()).add(payload["team2_id"])
        matches_by_game.append({"game": game, "matches": grouped})

    game_byes: dict[str, list[str]] = {}
    total_teams = {team.id: team.name for team in teams}
    for game, participants in game_participants.items():
        missing = [
            total_teams[team_id]
            for team_id in total_teams
            if team_id not in participants
        ]
        if missing:
            game_byes[game] = sorted(missing)

    leaderboard = _build_leaderboard(match_payload, team_lookup, bucket_pool_mode)
    playoff_match_obj = session.exec(
        select(Match)
        .where((Match.event_id == event.id) & (Match.is_playoff == True))
        .order_by(Match.order_index)
    ).first()

    group_stage_complete = all(match["status"] == "completed" for match in match_payload) if match_payload else False

    playoff_semifinals = session.exec(
        select(Match)
        .where(
            (Match.event_id == event.id)
            & (Match.is_playoff == True)
            & (Match.playoff_round == "semifinal")
        )
        .order_by(Match.order_index)
    ).all()

    semifinal_payload = [
        {
            "id": match.id,
            "team": team_lookup.get(match.team1_id, "TBD"),
            "team_id": match.team1_id,
            "score": match.score_team1,
            "status": match.status,
        }
        for match in playoff_semifinals
    ]
    semifinals_complete = bool(semifinal_payload) and all(item["score"] is not None for item in semifinal_payload)
    ordered_semis = sorted(
        semifinal_payload,
        key=lambda item: (item["score"] is None, item["score"] if item["score"] is not None else 9999),
    )
    semifinal_tie = False

    playoff_bracket = None
    playoff_match = None
    champion: dict[str, object] | None = None
    final_tie = False
    if playoff_match_obj:
        team1_name = team_lookup.get(playoff_match_obj.team1_id, "TBD")
        team2_name = team_lookup.get(playoff_match_obj.team2_id, "TBD")
        score1 = playoff_match_obj.score_team1
        score2 = playoff_match_obj.score_team2
        playoff_bracket = {
            "teams": [[team1_name, team2_name]],
                "results": [[[score1 if score1 is not None else None, score2 if score2 is not None else None]]],
        }
        playoff_match = {
            "id": playoff_match_obj.id,
            "team1": team1_name,
            "team2": team2_name,
            "score1": score1,
            "score2": score2,
            "status": playoff_match_obj.status,
        }
    else:
        if semifinals_complete and semifinal_payload:
            if len(ordered_semis) > 2 and ordered_semis[1]["score"] == ordered_semis[2]["score"]:
                semifinal_tie = True
            else:
                playoff_bracket = {
                    "teams": [[ordered_semis[0]["team"], ordered_semis[1]["team"]]],
                    "results": [[[None, None]]],
                }
        elif not semifinal_payload:
            seeds = [entry for entry in leaderboard if entry.get("bucket_score") is not None][:2]
            if len(seeds) == 2:
                playoff_bracket = {
                    "teams": [[seeds[0]["name"], seeds[1]["name"]]],
                    "results": [[[None, None]]],
                }

    if semifinal_payload:
        if semifinals_complete and not semifinal_tie:
            finalists_display = [
                {"name": entry["team"], "id": entry["team_id"]} for entry in ordered_semis[:2]
            ]
        else:
            finalists_display = []
    else:
        finalists_display = leaderboard[:2]

    if playoff_match_obj and playoff_match_obj.score_team1 is not None and playoff_match_obj.score_team2 is not None:
        score1 = playoff_match_obj.score_team1
        score2 = playoff_match_obj.score_team2
        if score1 == score2:
            final_tie = True
        else:
            game_name = playoff_match_obj.game.lower()
            lower_is_better = "bucket golf" in game_name and "beersbee" not in game_name
            if (lower_is_better and score1 < score2) or (not lower_is_better and score1 > score2):
                winner_id = playoff_match_obj.team1_id
                winner_score, loser_score = score1, score2
                runner_up_id = playoff_match_obj.team2_id
            else:
                winner_id = playoff_match_obj.team2_id
                winner_score, loser_score = score2, score1
                runner_up_id = playoff_match_obj.team1_id
            champion = {
                "name": team_lookup.get(winner_id, "TBD"),
                "scoreline": f"{winner_score} – {loser_score}",
                "runner_up": team_lookup.get(runner_up_id, "TBD"),
            }

    return {
        "event": event,
        "matches_by_game": matches_by_game,
        "team_count": len(team_lookup),
        "next_matches": next_matches,
        "leaderboard": leaderboard,
        "bucket_pool_mode": bucket_pool_mode,
        "top_four": leaderboard[:4],
        "finalists": finalists_display,
        "team_lookup": team_lookup,
        "playoff_match": playoff_match,
        "playoff_bracket": playoff_bracket,
        "playoff_semifinals": semifinal_payload,
        "playoff_semifinals_complete": semifinals_complete,
        "playoff_semifinals_tie": semifinal_tie,
        "champion": champion,
        "final_tie": final_tie,
        "group_stage_complete": group_stage_complete,
        "game_byes": game_byes,
    }


def _ensure_matches(
    session: Session, event: Event, bucket_pool_mode: bool, teams: list[Team] | None = None
) -> list[Match]:
    existing = session.exec(
        select(Match).where((Match.event_id == event.id) & (Match.is_playoff == False))
    ).all()
    if existing:
        return existing

    if teams is None:
        teams = session.exec(
            select(Team)
            .where((Team.event_id == event.id) & (Team.status == MODERATION_APPROVED))
            .order_by(Team.created_at)
        ).all()
    if len(teams) < 2:
        return []

    participants = [team.name for team in teams]
    name_to_team = {team.name: team for team in teams}
    id_to_name = {team.id: team.name for team in teams}
    team_ids = [team.id for team in teams]
    trio_mode = len(teams) == 3

    def select_slot(game_lists: dict[str, list[dict[str, str | None]]], game_order: list[str]) -> list[tuple[str, int]]:
        best: list[tuple[str, int]] = []
        best_key: tuple[int, int] = (0, 0)

        def backtrack(idx: int, used: set[int], picked: list[tuple[str, int]], idx_sum: int) -> None:
            nonlocal best, best_key
            if idx == len(game_order):
                key = (len(picked), -idx_sum)
                if key > best_key:
                    best_key = key
                    best = picked.copy()
                return

            game = game_order[idx]
            queue = game_lists.get(game, [])

            backtrack(idx + 1, used, picked, idx_sum)

            for position, matchup in enumerate(queue):
                team1 = name_to_team.get(matchup.get("team1")) if matchup.get("team1") else None
                team2 = name_to_team.get(matchup.get("team2")) if matchup.get("team2") else None
                if not team1:
                    continue
                ids = {team1.id}
                if team2 and team2.id != team1.id:
                    ids.add(team2.id)
                if ids & used:
                    continue
                picked.append((game, position))
                backtrack(idx + 1, used | ids, picked, idx_sum + position)
                picked.pop()

        backtrack(0, set(), [], 0)
        return best

    def build_slots(game_lists: dict[str, list[dict[str, str | None]]]) -> list[list[tuple[str, int, int]]]:
        lists = copy.deepcopy(game_lists)
        slots: list[list[tuple[str, int, int]]] = []
        safety_counter = 0
        max_iterations = sum(len(lists[game]) for game in GAMES) * 5 or 1
        game_order = list(GAMES)

        while any(lists[game] for game in GAMES) and safety_counter < max_iterations:
            safety_counter += 1
            selection = select_slot(lists, game_order)

            if not selection:
                fallback_game = next((game for game in GAMES if lists.get(game)), None)
                if fallback_game is None:
                    break
                selection = [(fallback_game, 0)]

            slot: list[tuple[str, int, int]] = []
            used_ids: set[int] = set()

            for game, index in sorted(selection, key=lambda item: item[1], reverse=True):
                queue = lists.get(game, [])
                if index >= len(queue):
                    continue
                matchup = queue.pop(index)
                team1 = name_to_team.get(matchup.get("team1")) if matchup.get("team1") else None
                team2 = name_to_team.get(matchup.get("team2")) if matchup.get("team2") else None
                if not team1:
                    continue
                team2_id = team1.id if not team2 else team2.id
                slot.append((game, team1.id, team2_id))
                used_ids.add(team1.id)
                if team2_id != team1.id:
                    used_ids.add(team2_id)

            if not slot:
                break

            for game in GAMES:
                max_open = MAX_OPEN_MATCHES_PER_GAME.get(game, 1)
                if max_open <= 1:
                    continue
                queue = lists.get(game, [])
                position = 0
                while position < len(queue) and sum(1 for g, _, _ in slot if g == game) < max_open:
                    matchup = queue[position]
                    team1 = name_to_team.get(matchup.get("team1")) if matchup.get("team1") else None
                    team2 = name_to_team.get(matchup.get("team2")) if matchup.get("team2") else None
                    if not team1:
                        queue.pop(position)
                        continue
                    ids = {team1.id}
                    if team2 and team2.id != team1.id:
                        ids.add(team2.id)
                    if ids & used_ids:
                        position += 1
                        continue
                    queue.pop(position)
                    team2_id = team1.id if not team2 else team2.id
                    slot.append((game, team1.id, team2_id))
                    used_ids.update(ids)

            slot.sort(key=lambda item: GAMES.index(item[0]))
            slots.append(slot)
            game_order = game_order[1:] + game_order[:1]

        return slots

    def build_candidate_sets(team_identifiers: list[int], required: int) -> list[tuple[tuple[int, int], ...]]:
        if required <= 0:
            return [tuple()]
        pairs = [
            (team_identifiers[i], team_identifiers[j])
            for i in range(len(team_identifiers))
            for j in range(i + 1, len(team_identifiers))
        ]
        candidates: list[tuple[tuple[int, int], ...]] = []
        for combo in itertools.combinations(pairs, required):
            usage: dict[int, int] = {}
            valid = True
            for team1_id, team2_id in combo:
                for identifier in (team1_id, team2_id):
                    usage[identifier] = usage.get(identifier, 0) + 1
                    if usage[identifier] > 1:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue
            if len(usage) != required * 2:
                continue
            candidates.append(combo)
        return candidates

    games_for_pairs = [game for game in GAMES if not (bucket_pool_mode and game == "Bucket Golf")]
    required_matches = (len(team_ids) // 2) if games_for_pairs else 0
    candidates_by_game: dict[str, list[tuple[tuple[int, int], ...]]] = {
        game: build_candidate_sets(team_ids, required_matches) for game in games_for_pairs
    }

    total_matches_needed = len(games_for_pairs) * required_matches
    unique_pairs_available = len(team_ids) * (len(team_ids) - 1) // 2
    duplicate_budget = max(0, total_matches_needed - unique_pairs_available)

    best_assignment: dict[str, tuple[tuple[int, int], ...]] | None = None
    best_score: tuple[int, int, int, int] | None = None

    if games_for_pairs and all(candidates_by_game.get(game) for game in games_for_pairs):
        def backtrack(
            index: int,
            pair_usage: dict[tuple[int, int], int],
            assignments: dict[str, tuple[tuple[int, int], ...]],
            per_team_totals: dict[int, int],
            per_game_penalty: int,
            duplicates_used: int,
            bye_counter: dict[int, int],
            bye_penalty: int,
        ) -> None:
            nonlocal best_assignment, best_score
            if index == len(games_for_pairs):
                values = list(per_team_totals.values())
                variance = (max(values) - min(values)) if values else 0
                score = (per_game_penalty, duplicates_used, variance, bye_penalty)
                if best_score is None or score < best_score:
                    best_score = score
                    best_assignment = assignments.copy()
                return

            game = games_for_pairs[index]
            for combo in candidates_by_game[game]:
                additional_duplicates = sum(1 for pair in combo if pair_usage.get(pair, 0) > 0)
                if duplicates_used + additional_duplicates > duplicate_budget:
                    continue

                updated_usage = pair_usage.copy()
                updated_totals = per_team_totals.copy()
                counts: defaultdict[int, int] = defaultdict(int)
                used_teams: set[int] = set()

                for team1_id, team2_id in combo:
                    pair = (team1_id, team2_id)
                    updated_usage[pair] = updated_usage.get(pair, 0) + 1
                    updated_totals[team1_id] += 1
                    updated_totals[team2_id] += 1
                    counts[team1_id] += 1
                    counts[team2_id] += 1
                    used_teams.add(team1_id)
                    used_teams.add(team2_id)

                penalty = sum(max(0, count - 1) for count in counts.values())
                bye_team = next((identifier for identifier in team_ids if identifier not in used_teams), None)
                updated_byes = bye_counter.copy()
                updated_bye_penalty = bye_penalty
                if bye_team is not None:
                    updated_byes[bye_team] = updated_byes.get(bye_team, 0) + 1
                    if updated_byes[bye_team] > 1:
                        updated_bye_penalty += 1

                assignments[game] = combo
                backtrack(
                    index + 1,
                    updated_usage,
                    assignments,
                    updated_totals,
                    per_game_penalty + penalty,
                    duplicates_used + additional_duplicates,
                    updated_byes,
                    updated_bye_penalty,
                )
                assignments.pop(game, None)

        initial_totals = {team_id: 0 for team_id in team_ids}
        backtrack(0, {}, {}, initial_totals, 0, 0, {}, 0)
        SCHEDULER_DEBUG["best_assignment"] = best_assignment
        SCHEDULER_DEBUG["best_score"] = best_score
    elif not games_for_pairs:
        best_assignment = {}

    if best_assignment is not None:
        game_lists: dict[str, list[dict[str, str | None]]] = {game: [] for game in GAMES}
        byes_by_game: dict[str, list[int]] = {}
        for game, combo in best_assignment.items():
            used: set[int] = set()
            for team1_id, team2_id in sorted(combo):
                game_lists[game].append(
                    {"team1": id_to_name[team1_id], "team2": id_to_name[team2_id]}
                )
                used.add(team1_id)
                used.add(team2_id)
            missing = [identifier for identifier in team_ids if identifier not in used]
            if missing:
                byes_by_game[game] = missing

        if len(team_ids) % 2 == 1:
            bye_candidates: list[int] = []
            for game in games_for_pairs:
                bye_candidates.extend(byes_by_game.get(game, []))
            unique_byes: list[int] = []
            for team_id in bye_candidates:
                if team_id not in unique_byes:
                    unique_byes.append(team_id)
                if len(unique_byes) == 2:
                    break
            if len(unique_byes) == 2:
                bye_team_a, bye_team_b = unique_byes
                game_lists.setdefault("KanJam", []).append(
                    {"team1": id_to_name[bye_team_a], "team2": id_to_name[bye_team_b]}
                )

        if bucket_pool_mode:
            game_lists["Bucket Golf"] = [
                {"team1": team.name, "team2": team.name} for team in teams
            ]

        slots = build_slots(game_lists)
        expected_matches = sum(len(entries) for entries in game_lists.values())
        planned_matches = sum(len(slot) for slot in slots)
        if slots and planned_matches == expected_matches:
            order = 1
            for slot in slots:
                for game, team1_id, team2_id in slot:
                    if team2_id == team1_id and game != "Bucket Golf":
                        continue
                    match = Match(
                        event_id=event.id,
                        game=game,
                        order_index=order,
                        team1_id=team1_id,
                        team2_id=team2_id,
                    )
                    session.add(match)
                    order += 1
            session.commit()
            return session.exec(select(Match).where(Match.event_id == event.id)).all()
        else:
            order = 1
            for game in GAMES:
                for entry in game_lists.get(game, []):
                    team1 = name_to_team.get(entry.get("team1"))
                    team2 = name_to_team.get(entry.get("team2")) if entry.get("team2") else None
                    if not team1:
                        continue
                    team2_id = team1.id if not team2 else team2.id
                    if team2_id == team1.id and game != "Bucket Golf":
                        continue
                    match = Match(
                        event_id=event.id,
                        game=game,
                        order_index=order,
                        team1_id=team1.id,
                        team2_id=team2_id,
                    )
                    session.add(match)
                    order += 1
            session.commit()
            return session.exec(select(Match).where(Match.event_id == event.id)).all()

    schedule = generate_schedule(participants)
    order = 1
    pending_matches: list[Match] = []

    for block in schedule:
        if block["game"] == "Bucket Golf" and bucket_pool_mode:
            for team in teams:
                pending_matches.append(
                    Match(
                        event_id=event.id,
                        game=block["game"],
                        order_index=order,
                        team1_id=team.id,
                        team2_id=team.id,
                    )
                )
                order += 1
            continue

        if trio_mode and block["game"] != "Bucket Golf":
            trio_pairs = [
                (teams[0], teams[1]),
                (teams[0], teams[2]),
                (teams[1], teams[2]),
            ]
            for team1, team2 in trio_pairs:
                pending_matches.append(
                    Match(
                        event_id=event.id,
                        game=block["game"],
                        order_index=order,
                        team1_id=team1.id,
                        team2_id=team2.id,
                    )
                )
                order += 1
            continue

        for matchup in block["matchups"]:
            team1_name = matchup.get("team1")
            team2_name = matchup.get("team2")
            if not team1_name or not team2_name:
                continue
            team1 = name_to_team.get(team1_name)
            team2 = name_to_team.get(team2_name)
            if not team1 or not team2:
                continue
            pending_matches.append(
                Match(
                    event_id=event.id,
                    game=block["game"],
                    order_index=order,
                    team1_id=team1.id,
                    team2_id=team2.id,
                )
            )
            order += 1

    for match in pending_matches:
        session.add(match)
    session.commit()

    return session.exec(select(Match).where(Match.event_id == event.id)).all()


def _refresh_match_statuses(session: Session, event_id: int) -> None:
    matches = session.exec(select(Match).where(Match.event_id == event_id).order_by(Match.order_index)).all()
    active_teams: set[int] = set()
    game_counts: dict[str, int] = defaultdict(int)

    for match in matches:
        if match.score_team1 is not None and match.score_team2 is not None:
            match.status = "completed"
            continue

        teams_in_match = {match.team1_id}
        if match.team2_id != match.team1_id:
            teams_in_match.add(match.team2_id)

        max_open = MAX_OPEN_MATCHES_PER_GAME.get(match.game, 1)
        if game_counts[match.game] < max_open and not (teams_in_match & active_teams):
            match.status = "in_progress"
            game_counts[match.game] += 1
            active_teams.update(teams_in_match)
        else:
            match.status = "pending"
    session.commit()


def _fetch_rsvps(session: Session, event_id: int, *, status: str | None = MODERATION_APPROVED):
    query = select(RSVP).where(RSVP.event_id == event_id)
    if status == MODERATION_APPROVED:
        query = query.where(RSVP.status == MODERATION_APPROVED)
    elif status == MODERATION_BLOCKED:
        query = query.where(RSVP.status == MODERATION_BLOCKED)
    return session.exec(query.order_by(RSVP.created_at.desc())).all()


def _fetch_photos(session: Session, event_id: int):
    return session.exec(select(Photo).where(Photo.event_id == event_id).order_by(Photo.created_at.desc())).all()


def _photo_image_url(filename: str) -> str:
    identifier = filename or ""
    if not identifier:
        return ""
    if identifier.startswith(("http://", "https://")):
        return identifier
    if is_gcs_identifier(identifier):
        object_name = extract_object_name(identifier)
        try:
            return gcs_public_url(object_name)
        except RuntimeError:
            base = os.getenv("GCS_PHOTO_BUCKET")
            if base:
                return f"https://storage.googleapis.com/{base}/{object_name}"
            return object_name
    return f"/static/uploads/{identifier}"


def _photo_payload(photo: Photo) -> dict[str, object]:
    return {
        "id": photo.id,
        "image_url": _photo_image_url(photo.filename),
        "created_at": photo.created_at,
        "original_name": photo.original_name,
        "event_id": photo.event_id,
    }


def _cast_match_payload(match: Match | None, team_lookup: dict[int, str]) -> dict[str, object] | None:
    if not match:
        return None

    team1_name = team_lookup.get(match.team1_id, "TBD")
    is_bye = match.team1_id == match.team2_id
    team2_name = None if is_bye else team_lookup.get(match.team2_id, "TBD")

    return {
        "id": match.id,
        "team1": team1_name,
        "team2": team2_name,
        "is_bye": is_bye,
        "status": match.status,
        "score1": match.score_team1,
        "score2": match.score_team2,
        "order": match.order_index,
    }


def _cast_state(session: Session) -> dict[str, object]:
    event = get_active_event(session)
    _refresh_match_statuses(session, event.id)

    teams = session.exec(
        select(Team)
        .where(Team.event_id == event.id)
        .order_by(Team.created_at)
    ).all()
    team_lookup = {team.id: team.name for team in teams}

    match_query = (
        select(Match)
        .where(Match.event_id == event.id)
        .order_by(Match.order_index)
    )
    matches = session.exec(match_query).all()

    grouped_matches: dict[str, list[Match]] = {}
    for match in matches:
        grouped_matches.setdefault(match.game, []).append(match)

    games_payload: list[dict[str, object]] = []
    for game_name, items in sorted(
        grouped_matches.items(), key=lambda entry: min(match.order_index for match in entry[1])
    ):
        items.sort(key=lambda match: match.order_index)
        current_match = next((match for match in items if match.status == "in_progress"), None)
        pending_matches = [
            match for match in items if match.status == "pending" and (not current_match or match.id != current_match.id)
        ]
        next_match = pending_matches[0] if pending_matches else None
        remaining_count = sum(1 for match in items if match.status in {"pending", "in_progress"})

        games_payload.append(
            {
                "game": game_name,
                "current": _cast_match_payload(current_match, team_lookup),
                "next": _cast_match_payload(next_match, team_lookup),
                "remaining": remaining_count,
                "upcoming_queue": [
                    _cast_match_payload(match, team_lookup) for match in pending_matches[1:3]
                ],
            }
        )

    photo_query = (
        select(Photo)
        .where(Photo.event_id == event.id)
        .order_by(Photo.created_at.desc())
        .limit(CAST_PHOTO_LIMIT)
    )
    photo_rows = session.exec(photo_query).all()
    photo_rows.reverse()

    photos_payload = [
        {
            "id": photo.id,
            "image_url": _photo_image_url(photo.filename),
            "created_at": photo.created_at.isoformat() if photo.created_at else None,
            "original_name": photo.original_name,
        }
        for photo in photo_rows
    ]

    return {
        "event": {
            "name": event.name,
            "slug": event.slug,
            "location": event.location,
            "event_date": event.event_date.isoformat() if event.event_date else None,
        },
        "generated_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "photos": photos_payload,
        "games": games_payload,
    }


def _team_context(
    *,
    session: Session,
    form_error: str | None,
    form_value: str,
    form_member_one: str,
    form_member_two: str,
    success_message: str | None,
    is_admin: bool,
) -> dict[str, object]:
    event = get_active_event(session)
    approved_teams = session.exec(
        select(Team)
        .where((Team.event_id == event.id) & (Team.status == MODERATION_APPROVED))
        .order_by(Team.created_at.desc())
    ).all()
    team_lookup = {team.id: team.name for team in approved_teams}
    free_agents_pending = _fetch_free_agents(session, event.id, status="pending")
    free_agents_paired = _fetch_free_agents(session, event.id, status="paired")

    blocked_teams: list[Team] = []
    blocked_pending_agents: list[FreeAgent] = []
    pending_team_updates: list[dict[str, object]] = []
    pending_free_agent_updates: list[dict[str, object]] = []
    if is_admin:
        blocked_teams = session.exec(
            select(Team)
            .where((Team.event_id == event.id) & (Team.status != MODERATION_APPROVED))
            .order_by(Team.created_at.desc())
        ).all()
        blocked_pending_agents = session.exec(
            select(FreeAgent)
            .where(
                (FreeAgent.event_id == event.id)
                & (FreeAgent.status == "pending")
                & (FreeAgent.moderation_status != MODERATION_APPROVED)
            )
            .order_by(FreeAgent.created_at.desc())
        ).all()
        pending_team_updates = _pending_updates(session, "team", Team)
        pending_free_agent_updates = _pending_updates(session, "freeagent", FreeAgent)

    return {
        "event": event,
        "teams": approved_teams,
        "form_error": form_error,
        "form_value": form_value,
        "form_member_one": form_member_one,
        "form_member_two": form_member_two,
        "success_message": success_message,
        "free_agents_pending": free_agents_pending,
        "free_agents_paired": free_agents_paired,
        "team_lookup": team_lookup,
        "blocked_teams": blocked_teams,
        "blocked_free_agents": blocked_pending_agents,
        "pending_team_updates": pending_team_updates,
        "pending_free_agent_updates": pending_free_agent_updates,
    }


def _fetch_free_agents(
    session: Session,
    event_id: int,
    *,
    status: str,
    include_blocked: bool = False,
) -> list[FreeAgent]:
    query = select(FreeAgent).where((FreeAgent.event_id == event_id) & (FreeAgent.status == status))
    if not include_blocked:
        query = query.where(FreeAgent.moderation_status == MODERATION_APPROVED)
    return session.exec(query.order_by(FreeAgent.created_at)).all()


def _pair_free_agents(session: Session, event: Event) -> list[Team]:
    pending = _fetch_free_agents(session, event.id, status="pending")
    created: list[Team] = []
    while len(pending) >= 2:
        first = pending.pop(0)
        second = pending.pop(0)
        team_name = _generate_free_agent_team_name(session, first, second)
        team = Team(
            name=team_name,
            event_id=event.id,
            member_one=(first.name.strip() or first.name),
            member_two=(second.name.strip() or second.name),
            status=MODERATION_APPROVED,
        )
        session.add(team)
        session.commit()
        session.refresh(team)

        first.status = "paired"
        first.team_id = team.id
        second.status = "paired"
        second.team_id = team.id
        session.add(first)
        session.add(second)
        session.commit()

        created.append(team)
        pending = _fetch_free_agents(session, event.id, status="pending")

    return created


def _generate_free_agent_team_name(session: Session, first: FreeAgent, second: FreeAgent) -> str:
    def short(name: str) -> str:
        token = name.split()[0] if name.strip() else "Agent"
        return token[: MAX_TEXT_LENGTH // 2]

    base = f"Free Agents {short(first.name)} & {short(second.name)}".strip()
    base = base[:MAX_TEXT_LENGTH]
    candidate = base
    counter = 1
    while session.exec(select(Team).where(Team.name == candidate)).first():
        suffix = f" #{counter}"
        candidate = (base[: max(1, MAX_TEXT_LENGTH - len(suffix))] + suffix).strip()
        counter += 1
    return candidate


def _clear_event_matches(session: Session, event_id: int, *, keep_playoffs: bool = False) -> None:
    query = select(Match).where(Match.event_id == event_id)
    if keep_playoffs:
        query = query.where(Match.is_playoff == False)
    matches = session.exec(query).all()
    for match in matches:
        session.delete(match)


def _team_lookup(session: Session, event_id: int) -> dict[int, str]:
    teams = session.exec(select(Team).where(Team.event_id == event_id)).all()
    return {team.id: team.name for team in teams}


def _build_leaderboard(
    matches: list[dict], team_lookup: dict[int, str], bucket_pool_mode: bool
) -> list[dict[str, object]]:
    stats: dict[int, dict[str, object]] = {}
    for team_id, name in team_lookup.items():
        stats[team_id] = {
            "name": name,
            "wins": 0,
            "ties": 0,
            "games": 0,
            "points_scored": 0,
            "points_allowed": 0,
        }

    bucket_scores: dict[int, int | None] = {team_id: None for team_id in team_lookup}

    for match in matches:
        team1_id = match["team1_id"]
        team2_id = match["team2_id"]
        score1 = match["score1"]
        score2 = match["score2"]
        game = match.get("game")
        if team1_id not in stats:
            stats[team1_id] = {
                "name": team_lookup.get(team1_id, "Team"),
                "wins": 0,
                "ties": 0,
                "games": 0,
                "points_scored": 0,
                "points_allowed": 0,
            }
        if team2_id not in stats:
            stats[team2_id] = {
                "name": team_lookup.get(team2_id, "Team"),
                "wins": 0,
                "ties": 0,
                "games": 0,
                "points_scored": 0,
                "points_allowed": 0,
            }
        if score1 is None or score2 is None:
            continue
        if team1_id == team2_id:
            stats[team1_id]["games"] += 1
        else:
            stats[team1_id]["games"] += 1
            stats[team2_id]["games"] += 1

        if game == "Bucket Golf":
            current = bucket_scores.get(team1_id)
            if current is None or score1 < current:
                bucket_scores[team1_id] = score1
            if team2_id != team1_id:
                current_two = bucket_scores.get(team2_id)
                if current_two is None or score2 < current_two:
                    bucket_scores[team2_id] = score2
            # Solo bucket runs skip head-to-head stats; wins are assigned after all scores are in.
            if team1_id == team2_id:
                continue

        stats[team1_id]["points_scored"] += score1
        stats[team1_id]["points_allowed"] += score2
        stats[team2_id]["points_scored"] += score2
        stats[team2_id]["points_allowed"] += score1

        if score1 > score2:
            stats[team1_id]["wins"] += 1
        elif score2 > score1:
            stats[team2_id]["wins"] += 1

    if bucket_pool_mode:
        completed_runs = [
            (team_id, score) for team_id, score in bucket_scores.items() if score is not None
        ]
        if completed_runs and len(completed_runs) == len(team_lookup):
            completed_runs.sort(key=lambda item: (item[1], stats[item[0]]["name"]))
            midpoint = len(completed_runs) // 2
            winners: set[int] = set()
            ties: set[int] = set()

            if len(completed_runs) % 2 == 0:
                for index, (team_id, _) in enumerate(completed_runs):
                    if index < midpoint:
                        winners.add(team_id)
                losers_start = midpoint
            else:
                for index, (team_id, _) in enumerate(completed_runs):
                    if index < midpoint:
                        winners.add(team_id)
                ties.add(completed_runs[midpoint][0])
                losers_start = midpoint + 1

            for team_id in winners:
                stats[team_id]["wins"] += 1
            for team_id in ties:
                stats[team_id]["ties"] += 1

    leaderboard = []
    for team_id, record in stats.items():
        losses = record["games"] - record["wins"] - record["ties"]
        entry = {
            "id": team_id,
            "name": record["name"],
            "wins": record["wins"],
            "ties": record["ties"],
            "losses": losses,
            "games": record["games"],
            "bucket_score": bucket_scores.get(team_id),
            "points_scored": record["points_scored"],
            "points_allowed": record["points_allowed"],
        }
        leaderboard.append(entry)

    def sort_key(item: dict[str, object]) -> tuple:
        wins = item["wins"]
        ties = item.get("ties", 0)
        games = item["games"]
        adjusted_wins = wins + 0.5 * ties
        win_pct = (adjusted_wins / games) if games else 0
        bucket_rank = item["bucket_score"] if item["bucket_score"] is not None else float("inf")
        point_diff = item["points_scored"] - item["points_allowed"]
        points_scored = item["points_scored"]
        return (-win_pct, bucket_rank, -point_diff, -points_scored, item["name"])

    leaderboard.sort(key=sort_key)
    return leaderboard


def _select_bucket_finalists(leaderboard: list[dict[str, object]]) -> tuple[list[dict[str, object]], str | None]:
    bucket_entries = [entry for entry in leaderboard if entry.get("bucket_score") is not None]
    if len(bucket_entries) < 2:
        return [], "need-scores"

    bucket_entries = sorted(bucket_entries, key=lambda entry: (entry["bucket_score"], entry["name"]))

    lowest_score = bucket_entries[0]["bucket_score"]
    first_group = [entry for entry in bucket_entries if entry["bucket_score"] == lowest_score]
    if len(first_group) > 1:
        return [], "tiebreak"

    first = bucket_entries[0]
    remaining = [entry for entry in bucket_entries if entry["id"] != first["id"]]
    if not remaining:
        return [], "need-scores"

    second_score = remaining[0]["bucket_score"]
    second_group = [entry for entry in remaining if entry["bucket_score"] == second_score]
    if len(second_group) > 1:
        return [], "tiebreak"

    return [first, second_group[0]], None
