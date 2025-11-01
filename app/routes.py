from __future__ import annotations

import hmac
import math
import os
import secrets
import shutil
import time
from pathlib import Path
from uuid import uuid4
from urllib.parse import quote, urljoin

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlmodel import Session, select

from .bracket import GAMES, generate_schedule

from .database import Event, FreeAgent, Match, Photo, RSVP, Team, UPLOAD_DIR, get_active_event, get_session

BASE_DIR = Path(__file__).resolve().parent

router = APIRouter()
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

ALLOWED_FREE_AGENT_STATUSES = {"pending", "paired"}

MAX_OPEN_MATCHES_PER_GAME = {
    "Bucket Golf": 8,
    "Bucket Golf Semifinal": 4,
}

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


def _render(
    request: Request,
    template_name: str,
    context: dict[str, object],
    *,
    status_code: int | None = None,
) -> HTMLResponse:
    payload = dict(context)
    payload["is_admin"] = _is_admin(request)
    response = templates.TemplateResponse(request, template_name, payload)
    if status_code is not None:
        response.status_code = status_code
    return response


@router.get("/", response_class=HTMLResponse, name="index")
async def index(request: Request, session: Session = Depends(get_session)):
    status = request.query_params.get("rsvp")
    context = _home_context(session, rsvp_status=status)
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
    username: str = Form(...),
    password: str = Form(...),
    next: str = Form(default="/bracket"),
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
async def admin_logout(request: Request, next: str | None = Form(default=None)):
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
    context = _photo_context(session, photo_success=success)
    return _render(request, "photos.html", context)


@router.get("/events", response_class=HTMLResponse, name="events_page")
async def events_page(request: Request, session: Session = Depends(get_session)):
    cards = _events_context(session)
    context = {"events": cards}
    return _render(request, "events.html", context)


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
        return _render(request, "index.html", context, status_code=400)

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


@router.post("/rsvp/{rsvp_id}/update", response_class=HTMLResponse, name="update_rsvp")
async def update_rsvp(
    request: Request,
    rsvp_id: int,
    session: Session = Depends(get_session),
    name: str = Form(...),
    guests: int = Form(...),
    message: str | None = Form(default=None),
):
    rsvp = session.get(RSVP, rsvp_id)
    if not rsvp:
        raise HTTPException(status_code=404, detail="RSVP not found")

    trimmed_name = name.strip()
    if not trimmed_name:
        context = _home_context(session, rsvp_error="Name is required.")
        return _render(request, "index.html", context, status_code=400)
    if guests < 1:
        context = _home_context(session, rsvp_error="Please include at least one guest.")
        return _render(request, "index.html", context, status_code=400)

    rsvp.name = trimmed_name
    rsvp.guests = guests
    rsvp.message = message.strip() if message else None
    session.add(rsvp)
    session.commit()

    redirect_url = str(request.url_for("index")) + "?rsvp=updated"
    return RedirectResponse(redirect_url, status_code=303)


@router.post("/rsvp/{rsvp_id}/delete", response_class=HTMLResponse, name="delete_rsvp")
async def delete_rsvp(request: Request, rsvp_id: int, session: Session = Depends(get_session)):
    rsvp = session.get(RSVP, rsvp_id)
    if not rsvp:
        raise HTTPException(status_code=404, detail="RSVP not found")
    session.delete(rsvp)
    session.commit()

    redirect_url = str(request.url_for("index")) + "?rsvp=deleted"
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
        return _render(request, "photos.html", context, status_code=400)

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
    form_error = None

    if request.query_params.get("created") == "1":
        success_message = "Team added successfully."

    context = _team_context(
        session=session,
        form_error=form_error,
        form_value="",
        form_member_one="",
        form_member_two="",
        success_message=success_message,
    )

    team_status = request.query_params.get("team")
    if team_status == "updated":
        context["success_message"] = "Team updated."
    elif team_status == "deleted":
        context["success_message"] = "Team removed."
    elif team_status == "exists":
        context["form_error"] = "That team name already exists."
    elif team_status == "invalid":
        context["form_error"] = "Team names must be at least two characters long."

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

    context.setdefault("free_agent_error", None)
    context.setdefault("free_agent_success", None)
    return _render(request, "teams.html", context)


@router.post("/teams", response_class=HTMLResponse, name="create_team")
async def create_team(
    request: Request,
    session: Session = Depends(get_session),
    name: str = Form(default=""),
    member_one: str = Form(default=""),
    member_two: str = Form(default=""),
):
    cleaned = name.strip()
    first_player = member_one.strip() or None
    second_player = member_two.strip() or None
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
            session=session,
            form_error=error,
            form_value=name,
            form_member_one=member_one,
            form_member_two=member_two,
            success_message=None,
        )
        return _render(request, "teams.html", context)

    team = Team(
        name=cleaned,
        event_id=event.id,
        member_one=first_player,
        member_two=second_player,
    )
    session.add(team)
    session.commit()
    session.refresh(team)

    _clear_event_matches(session, event.id)
    session.commit()

    redirect_url = str(request.url_for("team_directory")) + "?created=1"
    return RedirectResponse(redirect_url, status_code=303)


@router.post("/teams/{team_id}/update", response_class=HTMLResponse, name="update_team")
async def update_team(
    request: Request,
    team_id: int,
    session: Session = Depends(get_session),
    name: str = Form(...),
    member_one: str = Form(...),
    member_two: str = Form(...),
):
    team = session.get(Team, team_id)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")

    cleaned = name.strip()
    first_player = member_one.strip() or None
    second_player = member_two.strip() or None
    if len(cleaned) < 2:
        redirect_url = str(request.url_for("team_directory")) + "?team=invalid"
        return RedirectResponse(redirect_url, status_code=303)

    duplicate = session.exec(
        select(Team).where((Team.id != team.id) & (Team.event_id == team.event_id) & (Team.name == cleaned))
    ).first()
    if duplicate:
        redirect_url = str(request.url_for("team_directory")) + "?team=exists"
        return RedirectResponse(redirect_url, status_code=303)

    team.name = cleaned
    team.member_one = first_player
    team.member_two = second_player
    session.add(team)
    session.commit()

    redirect_url = str(request.url_for("team_directory")) + "?team=updated"
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

    _clear_event_matches(session, event_id)
    session.delete(team)
    session.commit()

    redirect_url = str(request.url_for("team_directory")) + "?team=deleted"
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
            session=session,
            form_error=None,
            form_value="",
            form_member_one="",
            form_member_two="",
            success_message=None,
        )
        context["free_agent_error"] = "Name is required for free agents."
        context.setdefault("free_agent_success", None)
        return _render(request, "teams.html", context, status_code=400)

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


@router.post("/free-agent/{agent_id}/update", response_class=HTMLResponse, name="update_free_agent")
async def update_free_agent(
    request: Request,
    agent_id: int,
    session: Session = Depends(get_session),
    name: str = Form(...),
    status: str = Form(...),
    note: str | None = Form(default=None),
    team_id: str | None = Form(default=None),
    pair_with: str | None = Form(default=None),
):
    agent = session.get(FreeAgent, agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Free agent not found")

    trimmed_name = name.strip()
    if not trimmed_name:
        redirect_url = str(request.url_for("team_directory")) + "?free_agent=error"
        return RedirectResponse(redirect_url, status_code=303)

    status_value = status.lower()
    if status_value not in ALLOWED_FREE_AGENT_STATUSES:
        status_value = "pending"

    agent.name = trimmed_name
    agent.note = note.strip() if note else None
    agent.status = status_value

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

    assigned_team_id: int | None = None
    if team_id:
        try:
            candidate_id = int(team_id)
        except ValueError:
            candidate_id = None
        if candidate_id is not None:
            team = session.get(Team, candidate_id)
            if team and team.event_id == agent.event_id:
                assigned_team_id = team.id

    if assigned_team_id is None or status_value == "pending":
        agent.team_id = None
        agent.status = "pending"
    else:
        agent.team_id = assigned_team_id
        agent.status = "paired"

    session.add(agent)
    session.commit()

    redirect_url = str(request.url_for("team_directory")) + "?free_agent=updated"
    return RedirectResponse(redirect_url, status_code=303)


@router.post("/free-agent/{agent_id}/delete", response_class=HTMLResponse, name="delete_free_agent")
async def delete_free_agent(request: Request, agent_id: int, session: Session = Depends(get_session)):
    agent = session.get(FreeAgent, agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Free agent not found")
    session.delete(agent)
    session.commit()

    redirect_url = str(request.url_for("team_directory")) + "?free_agent=deleted"
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
) -> dict[str, object]:
    event = get_active_event(session)
    rsvps = _fetch_rsvps(session, event.id)
    guest_total = sum(r.guests for r in rsvps)
    return {
        "event": event,
        "rsvps": rsvps,
        "guest_total": guest_total,
        "rsvp_error": rsvp_error,
        "rsvp_status": rsvp_status,
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


def _schedule_context(session: Session) -> dict[str, object]:
    event = get_active_event(session)
    teams = session.exec(select(Team).where(Team.event_id == event.id).order_by(Team.created_at)).all()
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
        teams = session.exec(select(Team).where(Team.event_id == event.id).order_by(Team.created_at)).all()
    if len(teams) < 2:
        return []

    participants = [team.name for team in teams]
    schedule = generate_schedule(participants)
    name_to_team = {team.name: team for team in teams}
    order = 1

    trio_mode = len(teams) == 3

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

        if trio_mode and block["game"] != "Bucket Golf":
            trio_pairs = [
                (teams[0], teams[1]),
                (teams[0], teams[2]),
                (teams[1], teams[2]),
            ]
            for team1, team2 in trio_pairs:
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

    for game_name, game_matches in by_game.items():
        max_open = MAX_OPEN_MATCHES_PER_GAME.get(game_name, 1)
        open_slots = 0
        for match in game_matches:
            if match.score_team1 is not None and match.score_team2 is not None:
                match.status = "completed"
                continue
            if open_slots < max_open:
                match.status = "in_progress"
                open_slots += 1
            else:
                match.status = "pending"
    session.commit()


def _fetch_rsvps(session: Session, event_id: int):
    return session.exec(select(RSVP).where(RSVP.event_id == event_id).order_by(RSVP.created_at.desc())).all()


def _fetch_photos(session: Session, event_id: int):
    return session.exec(select(Photo).where(Photo.event_id == event_id).order_by(Photo.created_at.desc())).all()


def _team_context(
    *,
    session: Session,
    form_error: str | None,
    form_value: str,
    form_member_one: str,
    form_member_two: str,
    success_message: str | None,
) -> dict[str, object]:
    event = get_active_event(session)
    teams = session.exec(select(Team).where(Team.event_id == event.id).order_by(Team.created_at.desc())).all()
    team_lookup = {team.id: team.name for team in teams}
    free_agents_pending = _fetch_free_agents(session, event.id, status="pending")
    free_agents_paired = _fetch_free_agents(session, event.id, status="paired")
    return {
        "event": event,
        "teams": teams,
        "form_error": form_error,
        "form_value": form_value,
        "form_member_one": form_member_one,
        "form_member_two": form_member_two,
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
        team = Team(
            name=team_name,
            event_id=event.id,
            member_one=(first.name.strip() or first.name),
            member_two=(second.name.strip() or second.name),
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
        return name.split()[0] if name.strip() else "Agent"

    base = f"Free Agents {short(first.name)} & {short(second.name)}"
    candidate = base
    counter = 1
    while session.exec(select(Team).where(Team.name == candidate)).first():
        counter += 1
        candidate = f"{base} #{counter}"
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
                "games": 0,
                "points_scored": 0,
                "points_allowed": 0,
            }
        if team2_id not in stats:
            stats[team2_id] = {
                "name": team_lookup.get(team2_id, "Team"),
                "wins": 0,
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
            # Solo bucket runs don't generate wins but still count as games.
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

    leaderboard = []
    for team_id, record in stats.items():
        entry = {
            "id": team_id,
            "name": record["name"],
            "wins": record["wins"],
            "games": record["games"],
            "bucket_score": bucket_scores.get(team_id),
            "points_scored": record["points_scored"],
            "points_allowed": record["points_allowed"],
        }
        leaderboard.append(entry)

    def sort_key(item: dict[str, object]) -> tuple:
        win_pct = (item["wins"] / item["games"]) if item["games"] else 0
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
