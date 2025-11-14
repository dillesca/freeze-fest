from __future__ import annotations

from collections import defaultdict
import math
from datetime import date

import pytest
from sqlmodel import SQLModel, Session, create_engine, select

from app.bracket import GAMES
from app.database import Event, Match, Team
from app.routes import (
    MAX_OPEN_MATCHES_PER_GAME,
    _build_leaderboard,
    _ensure_matches,
    _needs_bucket_pool,
    _group_bucket_pool_matches,
    _bucket_cast_sections,
)


def _build_test_engine():
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    return engine


def _reconstruct_slots(matches: list[Match]) -> list[list[Match]]:
    slots: list[list[Match]] = []
    current: list[Match] = []
    current_teams: set[int] = set()
    current_counts: defaultdict[str, int] = defaultdict(int)

    for match in sorted(matches, key=lambda obj: obj.order_index):
        teams_in_match = {match.team1_id}
        if match.team2_id != match.team1_id:
            teams_in_match.add(match.team2_id)

        capacity = MAX_OPEN_MATCHES_PER_GAME.get(match.game, 1)
        if teams_in_match & current_teams or current_counts[match.game] >= capacity:
            slots.append(current)
            current = []
            current_teams = set()
            current_counts = defaultdict(int)

        current.append(match)
        current_teams |= teams_in_match
        current_counts[match.game] += 1

    if current:
        slots.append(current)

    return slots


def _validate_schedule(team_count: int) -> None:
    engine = _build_test_engine()
    try:
        with Session(engine) as session:
            event = Event(
                name=f"Test Event {team_count}",
                slug=f"test-event-{team_count}",
                event_date=date(2025, 1, 1),
            )
            session.add(event)
            session.commit()
            session.refresh(event)
            event_id = event.id

            for index in range(team_count):
                session.add(Team(name=f"Team-{team_count}-{index}", event_id=event_id))
            session.commit()

            teams = session.exec(select(Team).where(Team.event_id == event_id)).all()

            bucket_pool = _needs_bucket_pool(team_count)
            _ensure_matches(session, event, bucket_pool, teams)
            session.commit()

        with Session(engine) as session:
            teams = session.exec(select(Team).where(Team.event_id == event_id)).all()
            matches = session.exec(
                select(Match).where(Match.event_id == event_id).order_by(Match.order_index)
            ).all()

            if bucket_pool:
                bucket_matches = [match for match in matches if match.game == "Bucket Golf"]
                assert bucket_matches, "Bucket Golf runs should be scheduled."
                assert all(
                    match.team1_id == match.team2_id for match in bucket_matches
                ), "Bucket Golf should be solo runs in pool mode."

            assert matches, "No matches were scheduled."

            # Ensure unique order indexes.
            order_indexes = [match.order_index for match in matches]
            assert len(order_indexes) == len(set(order_indexes))

            # Each team should play allotted games under the odd-team bye rules.
            appearances: dict[str, defaultdict[int, int]] = {game: defaultdict(int) for game in GAMES}
            match_counts: defaultdict[str, int] = defaultdict(int)
            for match in matches:
                match_counts[match.game] += 1
                appearances[match.game][match.team1_id] += 1
                if match.team2_id != match.team1_id:
                    appearances[match.game][match.team2_id] += 1

            bye_counts: defaultdict[str, int] = defaultdict(int)
            bye_games: defaultdict[int, set[str]] = defaultdict(set)
            for team in teams:
                for game in GAMES:
                    count = appearances[game].get(team.id, 0)
                    if game == "Bucket Golf" and bucket_pool:
                        assert count >= 1, f"{team.name} missing Bucket Golf solo run."
                        continue
                    if len(teams) % 2 == 1 and game != "Bucket Golf":
                        if count == 0:
                            bye_counts[game] += 1
                            bye_games[team.id].add(game)
                        continue
                    assert count >= 1, f"{team.name} missing {game} match."

            cornhole_bye_team: int | None = None

            if len(teams) % 2 == 1:
                for game in GAMES:
                    if game == "Bucket Golf":
                        continue
                    assert bye_counts[game] <= 1, f"Too many byes scheduled for {game}."
                assert bye_counts["Cornhole"] == 1, "Exactly one cornhole bye expected."
                cornhole_bye_team = next(
                    (team.id for team in teams if appearances["Cornhole"].get(team.id, 0) == 0),
                    None,
                )
                assert cornhole_bye_team is not None, "Cornhole bye team missing."
                assert (
                    appearances["KanJam"].get(cornhole_bye_team, 0) >= 2
                ), "Bye team should play an extra KanJam match."

            # Verify unique opponents per game (excluding solo runs).
            seen_pairs: set[tuple[str, int, int]] = set()
            for match in matches:
                if match.team1_id == match.team2_id:
                    continue
                key = (match.game, min(match.team1_id, match.team2_id), max(match.team1_id, match.team2_id))
                assert key not in seen_pairs, f"Duplicate pairing detected for {key}."
                seen_pairs.add(key)

            pair_counts_overall: defaultdict[tuple[int, int], int] = defaultdict(int)
            for match in matches:
                if match.team1_id == match.team2_id:
                    continue
                pair_key = (min(match.team1_id, match.team2_id), max(match.team1_id, match.team2_id))
                pair_counts_overall[pair_key] += 1

            active_games = [game for game in GAMES if not (bucket_pool and game == "Bucket Golf")]
            required_matches = math.ceil(len(teams) / 2) if active_games else 0
            total_matches_needed = len(active_games) * required_matches
            unique_pairs_available = len(teams) * (len(teams) - 1) // 2
            duplicate_budget = max(0, total_matches_needed - unique_pairs_available)
            actual_duplicates = sum(count - 1 for count in pair_counts_overall.values() if count > 1)
            assert (
                actual_duplicates <= duplicate_budget
            ), f"Excess duplicate opponents detected: {actual_duplicates} used, budget {duplicate_budget}."

            # Verify per-slot concurrency constraints.
            slots = _reconstruct_slots(matches)
            assert slots, "Slot reconstruction failed."

            for idx, slot in enumerate(slots):
                slot_teams: set[int] = set()
                game_counts: defaultdict[str, int] = defaultdict(int)

                for match in slot:
                    teams_in_match = {match.team1_id}
                    if match.team2_id != match.team1_id:
                        teams_in_match.add(match.team2_id)

                    assert not teams_in_match & slot_teams, (
                        f"Team scheduled in multiple games during slot {idx + 1}: {teams_in_match & slot_teams}"
                    )

                    slot_teams |= teams_in_match
                    game_counts[match.game] += 1
                    capacity = MAX_OPEN_MATCHES_PER_GAME.get(match.game, 1)
                    assert (
                        game_counts[match.game] <= capacity
                    ), f"{match.game} exceeds capacity in slot {idx + 1}"

                if bucket_pool:
                    continue

                # Check that each slot is maximal: no future match could be moved into this slot.
                future_matches = [m for future_slot in slots[idx + 1 :] for m in future_slot]
                for future in future_matches:
                    future_teams = {future.team1_id}
                    if future.team2_id != future.team1_id:
                        future_teams.add(future.team2_id)
                    capacity = MAX_OPEN_MATCHES_PER_GAME.get(future.game, 1)
                    if (
                        not future_teams & slot_teams
                        and game_counts[future.game] < capacity
                    ):
                        pytest.fail(
                            f"Match {future.game} ({future.team1_id} vs {future.team2_id}) "
                            f"could be played earlier in slot {idx + 1}."
                        )
    finally:
        engine.dispose()


@pytest.mark.parametrize("team_count", [3, 4, 5, 6, 7])
def test_scheduler_optimizes_concurrency(team_count: int):
    _validate_schedule(team_count)


def test_bucket_pool_assigns_wins_losses_and_ties():
    team_lookup = {team_id: f"Team {team_id}" for team_id in range(1, 6)}
    bucket_scores = {1: 12, 2: 14, 3: 18, 4: 22, 5: 30}

    matches = []
    for team_id, score in bucket_scores.items():
        matches.append(
            {
                "id": team_id,
                "game": "Bucket Golf",
                "team1_id": team_id,
                "team2_id": team_id,
                "score1": score,
                "score2": score,
            }
        )

    matches.append(
        {
            "id": 10,
            "game": "Cornhole",
            "team1_id": 1,
            "team2_id": 2,
            "score1": 11,
            "score2": 7,
        }
    )
    matches.append(
        {
            "id": 11,
            "game": "KanJam",
            "team1_id": 3,
            "team2_id": 4,
            "score1": 21,
            "score2": 12,
        }
    )

    leaderboard = _build_leaderboard(matches, team_lookup, bucket_pool_mode=True)
    records = {row["id"]: row for row in leaderboard}

    assert records[1]["wins"] == 2
    assert records[1]["losses"] == 0
    assert records[2]["wins"] == 1
    assert records[2]["losses"] == 1
    assert records[3]["ties"] == 1
    assert records[3]["wins"] == 1
    assert records[4]["losses"] == 2
    assert records[5]["losses"] == 1


def test_group_bucket_pool_matches_merges_runs():
    payloads = [
        {
            "id": 101,
            "team1": "Aurora",
            "team2": None,
            "is_bye": True,
            "status": "pending",
            "score1": None,
            "score2": None,
            "order": 1,
        },
        None,
        {
            "id": 102,
            "team1": "Borealis",
            "team2": None,
            "is_bye": True,
            "status": "pending",
            "score1": None,
            "score2": None,
            "order": 2,
        },
        {
            "id": 103,
            "team1": "Comet",
            "team2": None,
            "is_bye": True,
            "status": "pending",
            "score1": None,
            "score2": None,
            "order": 3,
        },
    ]

    grouped = _group_bucket_pool_matches(payloads)

    assert len(grouped) == 2
    assert grouped[0]["pool_group"] == ["Aurora", "Borealis"]
    assert grouped[0]["pool_pair"] is True
    assert grouped[0]["is_bye"] is False
    assert grouped[1]["pool_group"] == ["Comet"]
    assert grouped[1].get("pool_pair") is False


def test_group_bucket_pool_matches_creates_trio():
    payloads = [
        {"id": 201, "team1": "Aurora", "team2": None, "is_bye": True, "status": "pending", "score1": None, "score2": None, "order": 1},
        {"id": 202, "team1": "Borealis", "team2": None, "is_bye": True, "status": "pending", "score1": None, "score2": None, "order": 2},
        {"id": 203, "team1": "Comet", "team2": None, "is_bye": True, "status": "pending", "score1": None, "score2": None, "order": 3},
        {"id": 204, "team1": "Draco", "team2": None, "is_bye": True, "status": "pending", "score1": None, "score2": None, "order": 4},
        {"id": 205, "team1": "Equinox", "team2": None, "is_bye": True, "status": "pending", "score1": None, "score2": None, "order": 5},
    ]

    grouped = _group_bucket_pool_matches(payloads)

    assert len(grouped) == 2
    assert grouped[0]["pool_group"] == ["Aurora", "Borealis"]
    assert grouped[1]["pool_group"] == ["Comet", "Draco", "Equinox"]

    grouped_trio = _group_bucket_pool_matches(payloads[2:])
    assert len(grouped_trio) == 1
    assert grouped_trio[0]["pool_group"] == ["Comet", "Draco", "Equinox"]


def test_bucket_cast_sections_respects_wave_partners():
    team_lookup = {
        1: "Team1",
        2: "Team2",
        3: "Team3",
        4: "Team4",
        5: "Team5",
        6: "Team6",
        7: "Team7",
    }
    matches = [
        Match(id=1, event_id=1, game="Bucket Golf", order_index=1, team1_id=6, team2_id=6, status="in_progress"),
        Match(id=2, event_id=1, game="Bucket Golf", order_index=2, team1_id=2, team2_id=2, status="pending"),
        Match(id=3, event_id=1, game="Bucket Golf", order_index=3, team1_id=4, team2_id=4, status="pending"),
        Match(id=4, event_id=1, game="Bucket Golf", order_index=4, team1_id=1, team2_id=1, status="pending"),
        Match(id=5, event_id=1, game="Bucket Golf", order_index=5, team1_id=5, team2_id=5, status="pending"),
        Match(id=6, event_id=1, game="Bucket Golf", order_index=6, team1_id=7, team2_id=7, status="pending"),
        Match(id=7, event_id=1, game="Bucket Golf", order_index=7, team1_id=3, team2_id=3, status="pending"),
    ]

    current, next_payload, upcoming = _bucket_cast_sections(matches, team_lookup, True)

    assert current and current[0]["pool_group"] == ["Team6", "Team2"]
    assert next_payload and next_payload["pool_group"] == ["Team4", "Team1"]
    assert upcoming and upcoming[0]["pool_group"] == ["Team5", "Team7", "Team3"]


def test_bucket_cast_sections_handles_idle_state():
    team_lookup = {1: "Team1", 2: "Team2", 3: "Team3"}
    matches = [
        Match(id=1, event_id=1, game="Bucket Golf", order_index=1, team1_id=1, team2_id=1, status="pending"),
        Match(id=2, event_id=1, game="Bucket Golf", order_index=2, team1_id=2, team2_id=2, status="pending"),
        Match(id=3, event_id=1, game="Bucket Golf", order_index=3, team1_id=3, team2_id=3, status="pending"),
    ]

    current, next_payload, upcoming = _bucket_cast_sections(matches, team_lookup, True)

    assert current == []
    assert next_payload and next_payload["pool_group"] == ["Team1", "Team2"]
    assert upcoming and upcoming[0]["pool_group"] == ["Team3"]
