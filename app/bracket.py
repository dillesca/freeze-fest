"""Scheduling helpers for the three-game Freeze Fest tournament."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple, TypedDict

GAMES = ("Cornhole", "Bucket Golf", "Kanban")


class Matchup(TypedDict, total=False):
    team1: str | None
    team2: str | None
    note: str | None


class GameSchedule(TypedDict, total=False):
    game: str
    matchups: List[Matchup]


def normalize_participants(entries: Iterable[str]) -> List[str]:
    """Return participant names stripped of whitespace, ignoring blank rows."""
    return [value.strip() for value in entries if value and value.strip()]


def generate_schedule(participants: Sequence[str]) -> List[GameSchedule]:
    """Create pairings for each Freeze Fest game with minimal repeat opponents."""
    cleaned = normalize_participants(participants)
    if len(cleaned) < 2:
        cleaned = ["Team Aurora", "Team Borealis"]

    rounds = _round_robin_pairings(cleaned, len(GAMES))
    opponent_history = {team: set() for team in cleaned}

    schedule: List[GameSchedule] = []
    for index, game in enumerate(GAMES):
        pairings = rounds[index]
        matchups: List[Matchup] = []
        for team1, team2 in pairings:
            if team1 is None and team2 is None:
                matchups.append(Matchup(team1=None, team2=None, note="Open slot"))
                continue

            if team1 is None or team2 is None:
                solo = team1 or team2
                matchups.append(
                    Matchup(
                        team1=solo,
                        team2=None,
                        note=f"{solo} rests this round due to an odd number of teams.",
                    )
                )
                continue

            note = None
            if team2 in opponent_history[team1]:
                note = "Rematch required because of limited opponent combinations."
            opponent_history[team1].add(team2)
            opponent_history[team2].add(team1)
            matchups.append(Matchup(team1=team1, team2=team2, note=note))

        schedule.append(GameSchedule(game=game, matchups=matchups))

    return schedule


def _round_robin_pairings(teams: Sequence[str], round_count: int) -> List[List[Tuple[str | None, str | None]]]:
    """Return pairings for each round using the circle method rotation."""
    roster = list(teams)
    if len(roster) % 2 == 1:
        roster.append(None)

    if len(roster) <= 1:
        return [[(roster[0] if roster else None, None)] for _ in range(round_count)]

    working = roster[:]
    rounds: List[List[Tuple[str | None, str | None]]] = []
    for _ in range(round_count):
        pairs: List[Tuple[str | None, str | None]] = []
        for idx in range(len(working) // 2):
            pairs.append((working[idx], working[-(idx + 1)]))
        rounds.append(pairs)

        if len(working) <= 2:
            continue
        # Rotate all but the first position.
        working = [working[0]] + [working[-1]] + working[1:-1]

    return rounds
