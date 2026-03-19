"""MLB Stats API client.

Fetches regular-season game records from statsapi.mlb.com for each season.
Uses the schedule endpoint with linescore hydration to get runs, hits, errors,
and extra-innings flag in a single API call per season.

Saves compact game records (one dict per completed game) to data/raw/mlb_games_{year}.json.

Usage:
    python -m src.fetch.mlb_client --start 2011 --end 2024
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

RAW_DIR = Path(__file__).parents[2] / "data" / "raw"
BASE_URL = "https://statsapi.mlb.com/api/v1"
TEAMS_CACHE = RAW_DIR / "mlb_teams.json"


def _get(url: str, params: dict | None = None) -> dict:
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_team_abbreviations() -> dict[int, str]:
    """Return mapping of MLB team ID -> abbreviation. Cached locally."""
    if TEAMS_CACHE.exists():
        return {int(k): v for k, v in json.loads(TEAMS_CACHE.read_text()).items()}

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    data = _get(f"{BASE_URL}/teams", params={"sportId": 1})
    mapping = {
        t["id"]: t.get("abbreviation", t.get("teamCode", str(t["id"])))
        for t in data.get("teams", [])
    }
    TEAMS_CACHE.write_text(json.dumps(mapping))
    return mapping


def fetch_mlb_season(year: int) -> list[dict]:
    """Fetch all final regular-season games for a year.

    Returns a list of compact game dicts with runs, hits, errors, and innings played.
    """
    abbrevs = get_team_abbreviations()

    data = _get(
        f"{BASE_URL}/schedule",
        params={
            "sportId": 1,
            "season": year,
            "gameType": "R",
            "hydrate": "linescore",
        },
    )

    games = []
    for date_entry in data.get("dates", []):
        date_str = date_entry["date"]
        for game in date_entry.get("games", []):
            # Only include completed games
            if game.get("status", {}).get("detailedState") != "Final":
                continue

            game_pk = game["gamePk"]
            home_team = game["teams"]["home"]["team"]
            away_team = game["teams"]["away"]["team"]
            home_id = home_team["id"]
            away_id = away_team["id"]

            ls = game.get("linescore", {})
            ls_teams = ls.get("teams", {})

            # Prefer linescore for final stats; fall back to schedule score field
            home_runs = ls_teams.get("home", {}).get("runs")
            if home_runs is None:
                home_runs = game["teams"]["home"].get("score", 0)
            away_runs = ls_teams.get("away", {}).get("runs")
            if away_runs is None:
                away_runs = game["teams"]["away"].get("score", 0)

            home_hits = ls_teams.get("home", {}).get("hits", 0)
            away_hits = ls_teams.get("away", {}).get("hits", 0)
            home_errors = ls_teams.get("home", {}).get("errors", 0)
            away_errors = ls_teams.get("away", {}).get("errors", 0)
            innings = ls.get("currentInning", 9)

            games.append({
                "gamePk": game_pk,
                "date": date_str,
                "home_id": home_id,
                "home_abbrev": abbrevs.get(home_id, str(home_id)),
                "away_id": away_id,
                "away_abbrev": abbrevs.get(away_id, str(away_id)),
                "home_runs": int(home_runs or 0),
                "home_hits": int(home_hits or 0),
                "home_errors": int(home_errors or 0),
                "away_runs": int(away_runs or 0),
                "away_hits": int(away_hits or 0),
                "away_errors": int(away_errors or 0),
                "innings": int(innings or 9),
            })

    return games


def download_mlb_season(year: int, force: bool = False) -> Path | None:
    out = RAW_DIR / f"mlb_games_{year}.json"
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if out.exists() and not force:
        print(f"  {out.name} already exists, skipping")
        return out

    print(f"  Fetching MLB {year}...", end=" ", flush=True)
    try:
        games = fetch_mlb_season(year)
        out.write_text(json.dumps(games, indent=2))
        print(f"{len(games)} games -> {out.name}")
        time.sleep(0.5)  # be polite to the API
        return out
    except Exception as exc:
        print(f"ERROR: {exc}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MLB game data")
    parser.add_argument("--start", type=int, default=2011)
    parser.add_argument("--end", type=int, default=2024)
    parser.add_argument("--force", action="store_true", help="Re-download even if file exists")
    args = parser.parse_args()

    print(f"Downloading MLB seasons {args.start}–{args.end}")
    for year in range(args.start, args.end + 1):
        download_mlb_season(year, force=args.force)
    print("Done.")


if __name__ == "__main__":
    main()
