# Freeze Fest 2025 RSVP

A lightweight FastAPI web app for the 2025 Freeze Fest triathlon (November 15, 2025, 1:00 PM). Guests can browse event details, RSVP, upload photos, and manage teams. All data is stored in PostgreSQL (local Docker Compose or RDS/Aurora in AWS).

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
uvicorn app:app --reload --port 5000
```

Open http://127.0.0.1:5000 to interact with the site. Prefer containers for everything? Run `docker compose up --build` to launch the API + Postgres services together.

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
- Override `DATABASE_URL` per environment using `.env` or ECS task definitions.

## Project structure

- `app/` – FastAPI application package.
- `docker-compose.yml` / `.env.example` – local containers + env vars.
- `deploy/` – ECS task definition templates consumed by GitHub Actions.
- `.github/workflows/` – CI (`ci.yml`), production deploy (`deploy-prod.yml`), and the main-branch guard.
- `tests/` – pytest suite using an ephemeral SQLite DB.
- `requirements.txt`, `Dockerfile`, etc.

## CI/CD

- `.github/workflows/ci.yml` installs dependencies, compiles modules, and runs `python -m pytest` on every push/PR.
- `.github/workflows/deploy-prod.yml` builds/pushes the Docker image and updates the production ECS service on pushes (or manual dispatch) to `main`.
- `.github/workflows/main-branch-guard.yml` enforces the `ready-for-prod` label before PRs can merge into `main`.

## Branch strategy

1. Branch from `main` for every feature (`git checkout main && git pull && git checkout -b feature/foo`).
2. Open a pull request targeting `main` and keep pushing to the feature branch until CI is green and the review is approved (`ready-for-prod` label still gates the PR).
3. After approval, run the **Promote To Develop** workflow from the Actions tab and set `source_branch` to your feature branch. The workflow merges that branch into `develop`, which in turn triggers the normal CI run and the automatic dev deployment.
4. Once the dev environment is validated, trigger the **Promote To Main** workflow (source defaults to `develop`). That merge kicks off the production deployment workflow.
5. Delete the feature branch after promotion to keep history tidy.

## Required GitHub secrets

- `AWS_ROLE_ARN` – IAM role GitHub Actions assumes (via OIDC) for ECS/ECR operations.
- `ECR_REGISTRY` – e.g., `123456789012.dkr.ecr.us-east-1.amazonaws.com`.
- `ECR_REPOSITORY` – the ECR repo name (e.g., `freeze-fest`).

Customize `deploy/task-def.json` with your execution/task role ARNs, log group, env vars (including production `DATABASE_URL`), and ECS clusters/services (`freeze-fest-prod`, optional dev/staging if you add them back).
