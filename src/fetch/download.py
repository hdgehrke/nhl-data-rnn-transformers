"""CLI script to download NHL game data for a range of seasons.

Usage:
    python -m src.fetch.download --start 2011 --end 2024
    python -m src.fetch.download --start 2011 --end 2024 --force
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests

RAW_DIR = Path(__file__).parents[2] / "data" / "raw"
STATS_URL = "https://api.nhle.com/stats/rest/en"
SKIP_SEASONS = {"20042005"}


def season_str(year: int) -> str:
    return f"{year}{year + 1}"


def fetch_games_for_season(season: str) -> list[dict]:
    """Fetch all regular-season games for a season via the stats REST API."""
    url = f"{STATS_URL}/game"
    params = {
        "cayenneExp": f"season={season} and gameType=2",
        "limit": -1,
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", [])


def download_season(year: int, force: bool = False) -> Path | None:
    season = season_str(year)
    if season in SKIP_SEASONS:
        print(f"Skipping cancelled season {season}")
        return None

    out = RAW_DIR / f"games_{season}.json"
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if out.exists() and not force:
        print(f"  {out.name} already exists, skipping")
        return out

    print(f"  Fetching season {season}...", end=" ", flush=True)
    games = fetch_games_for_season(season)
    out.write_text(json.dumps(games, indent=2))
    print(f"{len(games)} games saved -> {out.name}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Download NHL game data")
    parser.add_argument("--start", type=int, default=2011, help="First season start year")
    parser.add_argument("--end", type=int, default=2024, help="Last season start year")
    parser.add_argument("--force", action="store_true", help="Re-download even if file exists")
    args = parser.parse_args()

    print(f"Downloading seasons {args.start}–{args.end}")
    for year in range(args.start, args.end + 1):
        download_season(year, force=args.force)
    print("Done.")


if __name__ == "__main__":
    main()
