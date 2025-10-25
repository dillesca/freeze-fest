# Freeze Fest 2025 RSVP

 A lightweight FastAPI web app for the 2025 Freeze Fest triathlon (November 15, 2025, 1:00 PM). Guests can read event details and RSVP so organizers know how many players to expect. All data is tied to the 2025 event in SQLite.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --port 5000
```

Open your browser at http://127.0.0.1:5000 to interact with the site.

## Managing teams

Visit `/teams` (Manage Teams link in the header) to add team names once. Teams are scoped to the 2025 event and stored in `freeze_fest.db` by default, or point the `DATABASE_URL` environment variable to another database supported by SQLModel. Free agents can also register without a partner; the app automatically pairs the next two and spins up a new team for them.

## RSVPs

Visitors can submit their name, email, party size, and any notes from the landing page. Responses are stored in the database and summarized in the guest list so you always know the expected attendance.

## Free agent pool

Individuals who don't yet have a teammate can register as free agents. The site keeps a waiting list, auto-matches the next two people, and creates a shared team so they appear in the bracket instantly.

## Photo gallery

Visit `/photos` to see the live slideshow. Attendees can upload PNG/JPG/GIF/WebP images from that page and the carousel updates immediately so everyone sees the latest highlights.

## Past events

Head to `/events` for the archive. Each event card shows the date, location, games played, winners (when available), and a mini photo gallery (including 2024's Cornhole/Kanban/Rillors tournament won by John & Stefan).

## Bracket preview

Use `/bracket` to show the automatically generated three-game rotation (Cornhole, Bucket Golf, Kanban). Teams can enter scores for the currently active match, the next game is highlighted, and a live leaderboard keeps track of wins.

### Playoffs

- Top four teams qualify based on the leaderboard.
- Playoff semifinal: single bucket-golf round with all four teams; lowest two scores advance (ties replay hole-by-hole).
- Championship: cornhole head-to-head between the two semifinal winners.

## 2025 event details

- **Date:** November 15, 2025
- **Location:** South Valley, Albuquerque, NM
- **Games:** Cornhole, Bucket Golf, Kanban
- **Slug:** `freeze-fest-2025`

All teams, schedules, and uploads are linked to this event record. If you need to reset the schema because the database existed before events were introduced, delete `freeze_fest.db` (or run your own migration) so SQLModel can add the new tables/columns.

## Docker

Build the container:

```bash
docker build -t freeze-fest .
```

Run it:

```bash
docker run --rm -p 5000:5000 freeze-fest
```

## Project structure

- `app/` – FastAPI application package
  - `bracket.py` – Scheduling helpers for the three-game rotation
  - `database.py` – SQLModel definitions for events, teams, free agents, matches, and photos
  - `routes.py` – HTTP routes and request handling
  - `templates/` – Jinja templates for rendering HTML
  - `static/` – CSS, JavaScript, and media uploads
- `deploy/` – ECS task definition templates used by GitHub Actions
- `.github/workflows/` – CI/CD pipelines for testing and deployments
- `Dockerfile` – Container recipe for deployment
- `requirements.txt` – Python dependencies pinned for reproducible builds

## Next steps

- Add authentication and role-based controls for admins vs. spectators
- Add optional scoring targets per game type
- Integrate WebSocket updates for live match progress
- Expose CSV exports for RSVPs, teams, and results

## CI/CD

- Pushes to `develop` trigger the Dev ECS deploy (`deploy-dev.yml`).
- Pushes to `staging` trigger the Staging ECS deploy (`deploy-staging.yml`).
- Pushes/dispatches on `main` trigger the Production ECS deploy (`deploy-prod.yml`).
- `.github/workflows/ci.yml` runs tests on every PR/push and should be required in branch protection rules.

### Required GitHub secrets

- `AWS_ROLE_ARN` – IAM role to assume via GitHub OIDC for ECR/ECS access.
- `ECR_REGISTRY` – e.g., `123456789012.dkr.ecr.us-east-1.amazonaws.com`.
- `ECR_REPOSITORY` – name of your ECR repo (e.g., `freeze-fest`).

Customize `deploy/task-def.json` with the correct task/execution roles, log group, env vars, etc., and provision ECS clusters/services named in the deploy workflows (`freeze-fest-dev`, `freeze-fest-staging`, `freeze-fest-prod`).
