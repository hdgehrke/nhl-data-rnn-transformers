"""NHL API client using the new api.nhle.com endpoints (post-2023 migration)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests

BASE_URL = "https://api-web.nhle.com/v1"
STATS_URL = "https://api.nhle.com/stats/rest/en"

# Seasons skipped or adjusted due to special circumstances
SKIP_SEASONS = {"20042005"}  # cancelled lockout season
SHORT_SEASONS = {"20122013", "20202021"}  # 48-game and 56-game seasons

RAW_DIR = Path(__file__).parents[2] / "data" / "raw"


def _get(url: str, params: dict | None = None, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def season_str(start_year: int) -> str:
    """Convert start year to NHL season string, e.g. 2011 -> '20112012'."""
    return f"{start_year}{start_year + 1}"


def fetch_schedule(season: str, save: bool = True) -> list[dict]:
    """Fetch full regular-season schedule for a given season string."""
    url = f"{BASE_URL}/club-schedule-season/NHL/{season}"
    # The new API uses team schedules; use the full schedule endpoint instead
    url = f"https://api-web.nhle.com/v1/schedule/{season[:4]}-10-01"

    # Fetch day by day via the full season schedule
    # Use the standings/schedule endpoint for complete season
    url = f"{STATS_URL}/game"
    params = {
        "cayenneExp": f"season={season} and gameType=2",
        "limit": -1,
    }
    data = _get(url, params=params)
    games = data.get("data", [])

    if save:
        out = RAW_DIR / f"schedule_{season}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(games, indent=2))
        print(f"Saved {len(games)} games for season {season} -> {out}")

    return games


def fetch_game_boxscore(game_id: int | str) -> dict:
    """Fetch boxscore for a single game."""
    url = f"{BASE_URL}/gamecenter/{game_id}/boxscore"
    return _get(url)


def fetch_game_landing(game_id: int | str) -> dict:
    """Fetch landing page data (includes advanced stats) for a single game."""
    url = f"{BASE_URL}/gamecenter/{game_id}/landing"
    return _get(url)


def fetch_team_stats_season(team_abbrev: str, season: str) -> dict:
    """Fetch aggregated team stats for a season."""
    url = f"{BASE_URL}/club-stats/{team_abbrev}/{season}/2"
    return _get(url)


def fetch_all_teams() -> list[dict]:
    """Return list of all NHL franchises."""
    url = f"{STATS_URL}/franchise"
    params = {"limit": -1}
    data = _get(url, params=params)
    return data.get("data", [])


def download_season(start_year: int, force: bool = False) -> Path:
    """Download full game log for a season to data/raw/. Returns path to JSON."""
    season = season_str(start_year)
    out = RAW_DIR / f"schedule_{season}.json"

    if out.exists() and not force:
        print(f"Already have {out}, skipping (use force=True to re-download)")
        return out

    if season in SKIP_SEASONS:
        print(f"Skipping cancelled season {season}")
        return out

    games = fetch_schedule(season, save=True)
    return out


def download_seasons(start: int, end: int, force: bool = False) -> list[Path]:
    """Download multiple seasons [start, end] inclusive by start year."""
    paths = []
    for year in range(start, end + 1):
        season = season_str(year)
        if season in SKIP_SEASONS:
            print(f"Skipping {season}")
            continue
        try:
            path = download_season(year, force=force)
            paths.append(path)
        except Exception as exc:
            print(f"ERROR fetching season {season}: {exc}")
    return paths


if __name__ == "__main__":
    # Quick smoke test
    print("Fetching teams...")
    teams = fetch_all_teams()
    print(f"Found {len(teams)} franchises")
