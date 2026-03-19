"""NBA API client using nba_api package.

Fetches regular-season game logs per season and saves as JSON to data/raw/.

Usage:
    python -m src.fetch.nba_client --start 2015 --end 2024
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from nba_api.stats.endpoints import leaguegamelog

RAW_DIR = Path(__file__).parents[2] / "data" / "raw"


def nba_season_str(start_year: int) -> str:
    """Convert start year to NBA season string, e.g. 2015 -> '2015-16'."""
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def fetch_nba_season(start_year: int, retries: int = 3) -> list[dict]:
    """Fetch all regular-season game log rows for a season."""
    season = nba_season_str(start_year)
    for attempt in range(retries):
        try:
            logs = leaguegamelog.LeagueGameLog(
                season=season,
                season_type_all_star="Regular Season",
                timeout=60,
            )
            df = logs.get_data_frames()[0]
            return df.to_dict(orient="records")
        except Exception as exc:
            if attempt == retries - 1:
                raise
            wait = 2 ** (attempt + 1)
            print(f"  Retry {attempt + 1}/{retries} after error: {exc} (waiting {wait}s)")
            time.sleep(wait)
    return []


def download_nba_season(start_year: int, force: bool = False) -> Path | None:
    out = RAW_DIR / f"nba_games_{nba_season_str(start_year)}.json"
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if out.exists() and not force:
        print(f"  {out.name} already exists, skipping")
        return out

    print(f"  Fetching NBA {nba_season_str(start_year)}...", end=" ", flush=True)
    try:
        records = fetch_nba_season(start_year)
        out.write_text(json.dumps(records, indent=2))
        # Count unique game IDs (each game has 2 rows, one per team)
        game_ids = {r["GAME_ID"] for r in records}
        print(f"{len(game_ids)} games ({len(records)} team-rows) -> {out.name}")
        time.sleep(1)  # be polite to the API
        return out
    except Exception as exc:
        print(f"ERROR: {exc}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Download NBA game data")
    parser.add_argument("--start", type=int, default=2015)
    parser.add_argument("--end", type=int, default=2024)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    print(f"Downloading NBA seasons {args.start}–{args.end}")
    for year in range(args.start, args.end + 1):
        download_nba_season(year, force=args.force)
    print("Done.")


if __name__ == "__main__":
    main()
