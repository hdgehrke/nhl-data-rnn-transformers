"""Parse raw NHL API JSON game records into GameToken objects and save as Parquet."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from src.features.schema import GameToken, FEATURE_NAMES, FEATURE_DIM

RAW_DIR = Path(__file__).parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"

# Expansion team first seasons — they have no prior history
EXPANSION_FIRST_SEASONS = {
    "VGK": "20172018",  # Vegas Golden Knights
    "SEA": "20212022",  # Seattle Kraken
}


def _parse_date(date_str: str) -> datetime:
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str[:10], "%Y-%m-%d")
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: {date_str}")


def _safe_int(val, default: int = 0) -> int:
    try:
        return int(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def parse_game_record(record: dict, season: str) -> tuple[GameToken, GameToken] | None:
    """Parse a single game record dict into (home_token, away_token).

    Returns None if the record is incomplete or should be skipped.
    The raw record format comes from: api.nhle.com/stats/rest/en/game
    """
    # Required fields
    game_id = str(record.get("id", ""))
    if not game_id:
        return None

    home_team = record.get("homeTeam", {})
    away_team = record.get("visitingTeam", {})
    if not home_team or not away_team:
        return None

    date_str = record.get("gameDate", "")
    if not date_str:
        return None

    home_score = _safe_int(record.get("homeScore"))
    away_score = _safe_int(record.get("visitingScore"))
    period_type = record.get("lastPeriodType", "REG")
    ot_game = 1 if period_type in ("OT", "SO") else 0

    home_win = 1 if home_score > away_score else 0
    away_win = 1 - home_win

    home_id = str(home_team.get("id", ""))
    away_id = str(away_team.get("id", ""))
    home_abbrev = home_team.get("abbrev", "")
    away_abbrev = away_team.get("abbrev", "")

    def make_token(is_home: bool) -> GameToken:
        gf = home_score if is_home else away_score
        ga = away_score if is_home else home_score
        win = home_win if is_home else away_win
        team_id = home_id if is_home else away_id
        team_abbrev = home_abbrev if is_home else away_abbrev
        opp_id = away_id if is_home else home_id
        opp_abbrev = away_abbrev if is_home else home_abbrev

        return GameToken(
            sport="nhl",
            season=season,
            game_id=game_id,
            team_id=team_id,
            team_abbrev=team_abbrev,
            opponent_id=opp_id,
            opponent_abbrev=opp_abbrev,
            date=date_str[:10],
            game_number=0,  # will be filled in by sequence builder
            goals_for=gf,
            goals_against=ga,
            win=win,
            ot_game=ot_game,
            is_home=1 if is_home else 0,
        )

    return make_token(True), make_token(False)


def tokens_to_dataframe(tokens: list[GameToken]) -> pd.DataFrame:
    """Convert a list of GameTokens to a DataFrame with feature columns."""
    records = []
    for tok in tokens:
        row = {
            "sport": tok.sport,
            "season": tok.season,
            "game_id": tok.game_id,
            "team_id": tok.team_id,
            "team_abbrev": tok.team_abbrev,
            "opponent_id": tok.opponent_id,
            "opponent_abbrev": tok.opponent_abbrev,
            "date": tok.date,
            "game_number": tok.game_number,
        }
        vec = tok.to_vector()
        for i, name in enumerate(FEATURE_NAMES):
            row[name] = vec[i]
        records.append(row)
    return pd.DataFrame(records)


def compute_rest_days(df: pd.DataFrame) -> pd.DataFrame:
    """Add rest_days and back_to_back columns, computed per team within season."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["team_id", "season", "date"])
    df["prev_date"] = df.groupby(["team_id", "season"])["date"].shift(1)
    df["rest_days"] = (df["date"] - df["prev_date"]).dt.days.fillna(3).clip(upper=7).astype(float)
    df["back_to_back"] = (df["rest_days"] == 1).astype(float)
    df["game_number"] = df.groupby(["team_id", "season"]).cumcount() + 1
    df = df.drop(columns=["prev_date"])
    return df


def process_season(season: str) -> pd.DataFrame:
    """Load raw JSON for a season, parse into tokens, return as DataFrame."""
    raw_path = RAW_DIR / f"games_{season}.json"
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found: {raw_path}")

    records = json.loads(raw_path.read_text())
    tokens: list[GameToken] = []
    skipped = 0
    for rec in records:
        result = parse_game_record(rec, season)
        if result is None:
            skipped += 1
            continue
        home_tok, away_tok = result
        tokens.extend([home_tok, away_tok])

    if skipped:
        print(f"  Skipped {skipped}/{len(records)} incomplete records")

    df = tokens_to_dataframe(tokens)
    df = compute_rest_days(df)
    return df


def save_processed(df: pd.DataFrame, season: str) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / f"games_{season}.parquet"
    df.to_parquet(out, index=False)
    print(f"  Saved {len(df)} rows -> {out.name}")
    return out


def process_all_seasons(seasons: list[str]) -> pd.DataFrame:
    """Process multiple seasons and return combined DataFrame."""
    dfs = []
    for season in seasons:
        print(f"Processing {season}...")
        try:
            df = process_season(season)
            save_processed(df, season)
            dfs.append(df)
        except FileNotFoundError as e:
            print(f"  Skipping: {e}")
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


if __name__ == "__main__":
    from src.fetch.download import season_str
    seasons = [season_str(y) for y in range(2011, 2025)]
    process_all_seasons(seasons)
