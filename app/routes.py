from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlmodel import Session, select

from .bracket import GAMES, generate_schedule
from .database import Event, FreeAgent, Match, Photo, RSVP, Team, UPLOAD_DIR, get_active_event, get_session

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

@router.get("/", response_class=HTMLResponse, name="index")
async def index(request: Request, session: Session = Depends(get_session)):
    success = request.query_params.get("rsvp") == "saved"
    context = _home_context(session, rsvp_success=success)
    context["request"] = request
    return templates.TemplateResponse("index.html", context)


@router.get("/bracket", response_class=HTMLResponse, name="bracket")
async def bracket_page(request: Request, session: Session = Depends(get_session)):
    context = _schedule_context(session)
    context["request"] = request
    return templates.TemplateResponse("bracket.html", context)


@router.get("/photos", response_class=HTMLResponse, name="photos")
async def photos_page(request: Request, session: Session = Depends(get_session)):
    success = request.query_params.get("photo") == "saved"
    context = _photo_context(session, photo_success=success)
    context["request"] = request
    return templates.TemplateResponse("photos.html", context)


@router.get("/events", response_class=HTMLResponse, name="events_page")
async def events_page(request: Request, session: Session = Depends(get_session)):
    cards = _events_context(session)
    context = {
        "request": request,
        "events": cards,
    }
    return templates.TemplateResponse("events.html", context)


@router.post("/rsvp", response_class=HTMLResponse, name="submit_rsvp")
async def submit_rsvp(
    request: Request,
    session: Session = Depends(get_session),
    name: str = Form(...),
    guests: int = Form(default=1),
    message: str | None = Form(default=None),
):
    event = get_active_event(session)
    trimmed_name = name.strip()
    error: str | None = None

    if not trimmed_name:
        error = "Name is required."
    elif guests < 1:
        error = "Please include at least one guest."

    if error:
        context = _home_context(session, rsvp_error=error)
        context["request"] = request
        return templates.TemplateResponse("index.html", context, status_code=400)

    rsvp = RSVP(
        name=trimmed_name,
        email="",
        guests=guests,
        message=message.strip() if message else None,
        event_id=event.id,
    )
    session.add(rsvp)
    session.commit()

    redirect_url = str(request.url_for("index")) + "?rsvp=saved"
    return RedirectResponse(redirect_url, status_code=303)


@router.post("/photos", response_class=HTMLResponse, name="upload_photo")
async def upload_photo(
    request: Request,
    session: Session = Depends(get_session),
    image: UploadFile = File(...),
):
    event = get_active_event(session)
    error: str | None = None
    original_name = image.filename or "upload.png"
    content_type = (image.content_type or "").lower()

    if not content_type.startswith("image/"):
        error = "Only image uploads are allowed."

    suffix = Path(original_name).suffix.lower() or ".png"
    allowed_suffixes = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
    if suffix not in allowed_suffixes:
        error = "Use PNG, JPG, GIF, or WebP images."

    if error:
        context = _photo_context(session, photo_error=error)
        context["request"] = request
        return templates.TemplateResponse("photos.html", context, status_code=400)

    filename = f"{uuid4().hex}{suffix}"
    destination = UPLOAD_DIR / filename
    image.file.seek(0)
    with destination.open("wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    photo = Photo(filename=filename, original_name=original_name, event_id=event.id)
    session.add(photo)
    session.commit()

    redirect_url = str(request.url_for("photos")) + "?photo=saved"
    return RedirectResponse(redirect_url, status_code=303)


@router.get("/teams", response_class=HTMLResponse, name="team_directory")
async def team_directory(request: Request, session: Session = Depends(get_session)):
    success_message = None
    if request.query_params.get("created") == "1":
        success_message = "Team added successfully."

    context = _team_context(
        request=request,
        session=session,
        form_error=None,
        form_value="",
        success_message=success_message,
    )
    flag = request.query_params.get("free_agent")
    if flag == "added":
        context["free_agent_success"] = "Added to the free-agent pool. We'll match you up soon."
    elif flag == "paired":
        context["free_agent_success"] = "Matched free agents and created a new team!"
    context.setdefault("free_agent_error", None)
    context.setdefault("free_agent_success", None)
    return templates.TemplateResponse("teams.html", context)


@router.post("/teams", response_class=HTMLResponse, name="create_team")
async def create_team(
    request: Request,
    session: Session = Depends(get_session),
    name: str = Form(default=""),
):
    cleaned = name.strip()
    error: str | None = None
    event = get_active_event(session)

    if len(cleaned) < 2:
        error = "Team names must be at least two characters long."
    else:
        existing = session.exec(select(Team).where((Team.name == cleaned) & (Team.event_id == event.id))).first()
        if existing:
            error = "That team already exists."

    if error:
        context = _team_context(
            request=request,
            session=session,
            form_error=error,
            form_value=name,
            success_message=None,
        )
        return templates.TemplateResponse("teams.html", context)

    team = Team(name=cleaned, event_id=event.id)
    session.add(team)
    session.commit()
    session.refresh(team)

    _clear_event_matches(session, event.id)

    redirect_url = str(request.url_for("team_directory")) + "?created=1"
    return RedirectResponse(redirect_url, status_code=303)


@router.post("/free-agent", response_class=HTMLResponse, name="register_free_agent")
async def register_free_agent(
    request: Request,
    session: Session = Depends(get_session),
    name: str = Form(...),
    email: str | None = Form(default=None),
    note: str | None = Form(default=None),
):
    trimmed_name = name.strip()
    trimmed_email = email.strip() if email else None
    event = get_active_event(session)

    if not trimmed_name:
        context = _team_context(
            request=request,
            session=session,
            form_error=None,
            form_value="",
            success_message=None,
        )
        context["free_agent_error"] = "Name is required for free agents."
        context.setdefault("free_agent_success", None)
        return templates.TemplateResponse("teams.html", context, status_code=400)

    agent = FreeAgent(
        name=trimmed_name,
        email=trimmed_email or "",
        note=note.strip() if note else None,
        event_id=event.id,
    )
    session.add(agent)
    session.commit()

    redirect_url = str(request.url_for("team_directory")) + "?free_agent=added"
    return RedirectResponse(redirect_url, status_code=303)


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
    rsvp_success: bool = False,
) -> dict[str, object]:
    event = get_active_event(session)
    rsvps = _fetch_rsvps(session, event.id)
    guest_total = sum(r.guests for r in rsvps)
    return {
        "event": event,
        "rsvps": rsvps,
        "guest_total": guest_total,
        "rsvp_error": rsvp_error,
        "rsvp_success": rsvp_success,
    }


def _photo_context(
    session: Session,
    *,
    photo_error: str | None = None,
    photo_success: bool = False,
) -> dict[str, object]:
    event = get_active_event(session)
    photos = _fetch_photos(session, event.id)
    return {
        "event": event,
        "photos": photos,
        "photo_error": photo_error,
        "photo_success": photo_success,
    }


def _events_context(session: Session) -> list[dict[str, object]]:
    events = session.exec(select(Event).order_by(Event.event_date.desc())).all()
    cards: list[dict[str, object]] = []
    for event in events:
        team_count = len(session.exec(select(Team).where(Team.event_id == event.id)).all())
        photos = session.exec(
            select(Photo)
            .where(Photo.event_id == event.id)
            .order_by(Photo.created_at.desc())
            .limit(4)
        ).all()
        cards.append({"event": event, "photos": photos, "team_count": team_count})
    return cards


def _needs_bucket_pool(team_count: int) -> bool:
    return team_count < len(GAMES) + 1


def _photo_context(
    session: Session,
    *,
    photo_error: str | None = None,
    photo_success: bool = False,
) -> dict[str, object]:
    event = get_active_event(session)
    photos = _fetch_photos(session, event.id)
    return {
        "event": event,
        "photos": photos,
        "photo_error": photo_error,
        "photo_success": photo_success,
    }


def _schedule_context(session: Session) -> dict[str, object]:
    event = get_active_event(session)
    teams = session.exec(select(Team).where(Team.event_id == event.id).order_by(Team.created_at)).all()
    bucket_pool_mode = _needs_bucket_pool(len(teams))
    matches = _ensure_matches(session, event, bucket_pool_mode, teams)
    _refresh_match_statuses(session, event.id)
    matches = session.exec(select(Match).where(Match.event_id == event.id).order_by(Match.order_index)).all()
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
    for game in GAMES:
        grouped = [payload for payload in match_payload if payload["game"] == game]
        if not grouped:
            continue
        # Determine next match for this game.
        next_match = next((payload for payload in grouped if payload["status"] != "completed"), None)
        if next_match:
            next_matches.append({"game": game, "match": next_match})
        matches_by_game.append({"game": game, "matches": grouped})

    leaderboard = _build_leaderboard(match_payload, team_lookup, bucket_pool_mode)

    return {
        "event": event,
        "matches_by_game": matches_by_game,
        "team_count": len(team_lookup),
        "next_matches": next_matches,
        "leaderboard": leaderboard,
        "bucket_pool_mode": bucket_pool_mode,
        "top_four": leaderboard[:4],
        "finalists": leaderboard[:2],
    }


def _ensure_matches(
    session: Session, event: Event, bucket_pool_mode: bool, teams: list[Team] | None = None
) -> list[Match]:
    existing = session.exec(select(Match).where(Match.event_id == event.id)).all()
    if existing:
        return existing

    if teams is None:
        teams = session.exec(select(Team).where(Team.event_id == event.id).order_by(Team.created_at)).all()
    if len(teams) < 2:
        return []

    participants = [team.name for team in teams]
    schedule = generate_schedule(participants)
    name_to_team = {team.name: team for team in teams}
    order = 1

    for block in schedule:
        if block["game"] == "Bucket Golf" and bucket_pool_mode:
            for team in teams:
                match = Match(
                    event_id=event.id,
                    game=block["game"],
                    order_index=order,
                    team1_id=team.id,
                    team2_id=team.id,
                )
                session.add(match)
                session.commit()
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
            match = Match(
                event_id=event.id,
                game=block["game"],
                order_index=order,
                team1_id=team1.id,
                team2_id=team2.id,
            )
            session.add(match)
            session.commit()
            order += 1

    return session.exec(select(Match).where(Match.event_id == event.id)).all()


def _refresh_match_statuses(session: Session, event_id: int) -> None:
    matches = session.exec(select(Match).where(Match.event_id == event_id).order_by(Match.order_index)).all()
    by_game: dict[str, list[Match]] = {}
    for match in matches:
        by_game.setdefault(match.game, []).append(match)

    for game_matches in by_game.values():
        in_progress_set = False
        for match in game_matches:
            if match.score_team1 is not None and match.score_team2 is not None:
                match.status = "completed"
                continue
            if not in_progress_set:
                match.status = "in_progress"
                in_progress_set = True
            else:
                match.status = "pending"
    session.commit()


def _fetch_rsvps(session: Session, event_id: int):
    return session.exec(select(RSVP).where(RSVP.event_id == event_id).order_by(RSVP.created_at.desc())).all()


def _fetch_photos(session: Session, event_id: int):
    return session.exec(select(Photo).where(Photo.event_id == event_id).order_by(Photo.created_at.desc())).all()


def _team_context(
    *,
    request: Request,
    session: Session,
    form_error: str | None,
    form_value: str,
    success_message: str | None,
) -> dict[str, object]:
    event = get_active_event(session)
    teams = session.exec(select(Team).where(Team.event_id == event.id).order_by(Team.created_at.desc())).all()
    team_lookup = {team.id: team.name for team in teams}
    free_agents_pending = _fetch_free_agents(session, event.id, status="pending")
    free_agents_paired = _fetch_free_agents(session, event.id, status="paired")
    return {
        "request": request,
        "event": event,
        "teams": teams,
        "form_error": form_error,
        "form_value": form_value,
        "success_message": success_message,
        "free_agents_pending": free_agents_pending,
        "free_agents_paired": free_agents_paired,
        "team_lookup": team_lookup,
    }


def _fetch_free_agents(session: Session, event_id: int, *, status: str) -> list[FreeAgent]:
    return session.exec(
        select(FreeAgent)
        .where((FreeAgent.event_id == event_id) & (FreeAgent.status == status))
        .order_by(FreeAgent.created_at)
    ).all()


def _pair_free_agents(session: Session, event: Event) -> list[Team]:
    pending = _fetch_free_agents(session, event.id, status="pending")
    created: list[Team] = []
    while len(pending) >= 2:
        first = pending.pop(0)
        second = pending.pop(0)
        team_name = _generate_free_agent_team_name(session, first, second)
        team = Team(name=team_name, event_id=event.id)
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
        return name.split()[0] if name.strip() else "Agent"

    base = f"Free Agents {short(first.name)} & {short(second.name)}"
    candidate = base
    counter = 1
    while session.exec(select(Team).where(Team.name == candidate)).first():
        counter += 1
        candidate = f"{base} #{counter}"
    return candidate


def _clear_event_matches(session: Session, event_id: int) -> None:
    matches = session.exec(select(Match).where(Match.event_id == event_id)).all()
    for match in matches:
        session.delete(match)
    session.commit()


def _team_lookup(session: Session, event_id: int) -> dict[int, str]:
    teams = session.exec(select(Team).where(Team.event_id == event_id)).all()
    return {team.id: team.name for team in teams}


def _build_leaderboard(
    matches: list[dict], team_lookup: dict[int, str], bucket_pool_mode: bool
) -> list[dict[str, object]]:
    stats: dict[int, dict[str, object]] = {}
    for team_id, name in team_lookup.items():
        stats[team_id] = {"name": name, "wins": 0, "games": 0}

    bucket_scores: dict[int, int | None] = {team_id: None for team_id in team_lookup}

    for match in matches:
        team1_id = match["team1_id"]
        team2_id = match["team2_id"]
        score1 = match["score1"]
        score2 = match["score2"]
        game = match.get("game")
        if team1_id not in stats:
            stats[team1_id] = {"name": team_lookup.get(team1_id, "Team"), "wins": 0, "games": 0}
        if team2_id not in stats:
            stats[team2_id] = {"name": team_lookup.get(team2_id, "Team"), "wins": 0, "games": 0}
        if score1 is None or score2 is None:
            continue
        if team1_id == team2_id:
            stats[team1_id]["games"] += 1
        else:
            stats[team1_id]["games"] += 1
            stats[team2_id]["games"] += 1

        if bucket_pool_mode and game == "Bucket Golf" and team1_id == team2_id:
            current = bucket_scores.get(team1_id)
            if current is None or score1 < current:
                bucket_scores[team1_id] = score1
            continue

        if bucket_pool_mode and game != "Bucket Golf":
            continue

        if game == "Bucket Golf":
            if score1 < score2:
                stats[team1_id]["wins"] += 1
            elif score2 < score1:
                stats[team2_id]["wins"] += 1
        else:
            if score1 > score2:
                stats[team1_id]["wins"] += 1
            elif score2 > score1:
                stats[team2_id]["wins"] += 1

    leaderboard = []
    for team_id, record in stats.items():
        entry = {
            "name": record["name"],
            "wins": record["wins"],
            "games": record["games"],
            "bucket_score": bucket_scores.get(team_id),
        }
        leaderboard.append(entry)

    if bucket_pool_mode:
        leaderboard.sort(key=lambda item: (item["bucket_score"] is None, item["bucket_score"] or 9999))
    else:
        leaderboard.sort(key=lambda item: item["wins"], reverse=True)
    return leaderboard
