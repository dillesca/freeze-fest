# Freeze Fest 2025 RSVP

A lightweight FastAPI web app for the 2025 Freeze Fest triathlon (November 15, 2025, 1:00 PM). Guests can browse event details, RSVP, upload photos, and manage teams. All data is stored in PostgreSQL (local Docker Compose or Cloud SQL in Google Cloud).

## Quick start

```bash
# local env vars + docker compose defaults
cp .env.example .env

# start a Postgres instance for local dev
docker compose up -d postgres

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export DATABASE_URL=postgresql://freeze_fest:freeze_fest@localhost:5432/freeze_fest
uvicorn app:app --reload --port 8080
```

Open http://127.0.0.1:8080 to interact with the site. Prefer containers for everything? Run `docker compose up --build` to launch the API + Postgres services together. The containerized API is exposed on http://localhost:8080.

## Running tests

```bash
# local venv
python -m pytest

# inside Docker (matches CI image)
docker compose build api
docker compose run --rm api python -m pytest
```

The tests point `DATABASE_URL` at a temporary SQLite file to stay fast and isolated from your Postgres instances; the override happens automatically inside the test harness.

## Features

- **RSVP / Teams** – submits attendees, collects free agents, and stores teams per event.
- **Photo uploads** – `/photos` page lets attendees upload images (stored per event).
- **Past events** – `/events` lists previous Freeze Fest tournaments, winners, and photos.
- **Bracket & playoffs** – `/bracket` shows live match rotation, scoring, and playoff picture.

## Playoffs format

1. Top four leaderboard teams enter a bucket-golf semifinal (lowest two scores advance, ties replay a hole).  
2. Final two teams play a cornhole championship match.

## Local Docker Compose

- `.env.example` defines defaults for the Postgres service (`POSTGRES_USER/PASSWORD/DB`).
- `docker compose up -d postgres` starts the DB only; `docker compose up --build` runs API + DB together.
- Override `DATABASE_URL` per environment using `.env` or deployment-time configuration (e.g., Cloud Run secrets or env vars).

## Project structure

- `app/` – FastAPI application package.
- `docker-compose.yml` / `.env.example` – local containers + env vars.
- `.github/workflows/` – CI (`ci.yml`) and the main-branch guard workflow.
- `tests/` – pytest suite using an ephemeral SQLite DB.
- `requirements.txt`, `Dockerfile`, etc.

## CI/CD

- `.github/workflows/ci.yml` installs dependencies, compiles modules, and runs `python -m pytest` on every push/PR.
- `.github/workflows/main-branch-guard.yml` enforces the `ready-for-prod` label before PRs can merge into `main`.
- Cloud Run deployments are triggered manually via `gcloud run deploy` (or via Cloud Build triggers) using the image published to Artifact Registry.

## Branch strategy

1. Branch from `main` for every feature (`git checkout main && git pull && git checkout -b feature/foo`).
2. Open a pull request targeting `main` and keep pushing to the feature branch until CI is green and the review is approved (`ready-for-prod` label still gates the PR).
3. Run the **Promote To Develop** workflow (Actions tab) after approval to keep `develop` current. This branch is our staging ground but does not deploy anywhere automatically.
4. Merge `develop` into `main` via the **Promote To Main** workflow once the change is validated. The `main` branch is the source of truth for production, and each merge should be followed by a Cloud Run deploy (`gcloud run deploy` using the commands below).
5. Delete the feature branch after promotion to keep history tidy.

## Deployment (Cloud Run)

`main` represents the version currently deployed to Google Cloud. After merging into `main`, build and push the container image to Artifact Registry, then deploy:

```bash
gcloud builds submit --tag us-west1-docker.pkg.dev/<PROJECT_ID>/freeze-fest/freeze-fest:latest

gcloud run deploy freeze-fest \
  --image us-west1-docker.pkg.dev/<PROJECT_ID>/freeze-fest/freeze-fest:latest \
  --region us-west1 \
  --service-account freeze-fest-sa@<PROJECT_ID>.iam.gserviceaccount.com \
  --add-cloudsql-instances <PROJECT_ID>:us-west1:freeze-fest-pg \
  --set-env-vars DATABASE_URL="postgresql+psycopg2://freeze:<PASSWORD>@/freeze_db?host=/cloudsql/<PROJECT_ID>:us-west1:freeze-fest-pg"
```

Replace the placeholders with your project ID, database password, and Cloud SQL instance details. Map `DATABASE_URL` from Secret Manager if you prefer not to inline credentials.
