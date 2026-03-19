"""Parse raw MLB game JSON into GameToken objects and save as Parquet.

Feature mapping from MLB box score stats to the shared GameToken schema:

  Shared (populated):
    goals_for       <- runs_scored (team's runs in the game)
    goals_against   <- runs_allowed
    goal_diff       <- computed
    win             <- runs_scored > runs_allowed
    ot_game         <- extra_innings flag (innings > 9)
    shots_for       <- hits_for (hits by team's batters)
    shots_against   <- hits_against (hits allowed by pitchers)
    giveaways       <- errors (fielding errors)
    is_home         <- home/away flag
    rest_days       <- computed from game dates (capped at 7)
    back_to_back    <- rest_days == 1
    game_number     <- cumulative game count within season

  Note: shooting_pct and save_pct are derived properties of GameToken
  (goals/shots and saved/shots), so they capture runs-per-hit conversion
  rate rather than traditional batting average.

  Zero-filled (no direct MLB equivalent):
    hits, blocks, takeaways, pim
    pp_pct, pk_pct, pp_goals_*, pp_opps_*
    corsi_for_pct, fenwick_for_pct, xgoals_for, xgoals_against

Usage:
    python -m src.features.mlb_tokenizer [--start 2011] [--end 2024]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.features.schema import FEATURE_NAMES, GameToken

RAW_DIR = Path(__file__).parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"


def _safe_int(val, default: int = 0) -> int:
    try:
        return int(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def parse_mlb_season(year: int) -> pd.DataFrame:
    """Load MLB game JSON for a year and return processed DataFrame."""
    raw_path = RAW_DIR / f"mlb_games_{year}.json"
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw MLB data not found: {raw_path}")

    games = json.loads(raw_path.read_text())
    if not games:
        return pd.DataFrame()

    # Expand each game into two team-perspective rows (home and away)
    rows = []
    for g in games:
        for side, opp_side in (("home", "away"), ("away", "home")):
            rows.append({
                "game_id": str(g["gamePk"]),
                "date": g["date"],
                "team_id": str(g[f"{side}_id"]),
                "team_abbrev": g[f"{side}_abbrev"],
                "opponent_id": str(g[f"{opp_side}_id"]),
                "opponent_abbrev": g[f"{opp_side}_abbrev"],
                "runs_for": g[f"{side}_runs"],
                "runs_against": g[f"{opp_side}_runs"],
                "hits_for": g[f"{side}_hits"],
                "hits_against": g[f"{opp_side}_hits"],
                "errors": g[f"{side}_errors"],
                "innings": g["innings"],
                "is_home": 1 if side == "home" else 0,
            })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["team_id", "date"]).reset_index(drop=True)

    # Compute context features per team
    df["prev_date"] = df.groupby("team_id")["date"].shift(1)
    df["rest_days"] = (
        (df["date"] - df["prev_date"]).dt.days.fillna(3).clip(upper=7).astype(int)
    )
    df["back_to_back"] = (df["rest_days"] == 1).astype(int)
    df["game_number"] = df.groupby("team_id").cumcount() + 1

    # Build GameTokens and convert to feature vectors
    output_rows = []
    for _, row in df.iterrows():
        runs_for = _safe_int(row["runs_for"])
        runs_against = _safe_int(row["runs_against"])

        tok = GameToken(
            sport="mlb",
            season=str(year),
            game_id=row["game_id"],
            team_id=row["team_id"],
            team_abbrev=row["team_abbrev"],
            opponent_id=row["opponent_id"],
            opponent_abbrev=row["opponent_abbrev"],
            date=str(row["date"].date()),
            game_number=int(row["game_number"]),
            goals_for=runs_for,
            goals_against=runs_against,
            win=1 if runs_for > runs_against else 0,
            ot_game=1 if _safe_int(row["innings"]) > 9 else 0,
            shots_for=_safe_int(row["hits_for"]),
            shots_against=_safe_int(row["hits_against"]),
            hits=0,
            blocks=0,
            giveaways=_safe_int(row["errors"]),
            takeaways=0,
            pim=0,
            pp_goals_for=0,
            pp_goals_against=0,
            pp_opps_for=0,
            pp_opps_against=0,
            corsi_for_pct=0.0,
            fenwick_for_pct=0.0,
            xgoals_for=0.0,
            xgoals_against=0.0,
            is_home=int(row["is_home"]),
            rest_days=int(row["rest_days"]),
            back_to_back=int(row["back_to_back"]),
        )

        row_dict = {
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
            row_dict[name] = float(vec[i])
        output_rows.append(row_dict)

    return pd.DataFrame(output_rows)


def process_mlb_season(year: int) -> pd.DataFrame:
    print(f"Processing MLB {year}...")
    df = parse_mlb_season(year)
    if df.empty:
        print("  Empty — skipping")
        return df
    out = PROCESSED_DIR / f"mlb_games_{year}.parquet"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    n_games = df["game_id"].nunique()
    print(f"  {n_games} games ({len(df)} team-rows) -> {out.name}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Process raw MLB data into Parquet")
    parser.add_argument("--start", type=int, default=2011)
    parser.add_argument("--end", type=int, default=2024)
    args = parser.parse_args()

    for year in range(args.start, args.end + 1):
        try:
            process_mlb_season(year)
        except FileNotFoundError as e:
            print(f"  Skipping: {e}")


if __name__ == "__main__":
    main()
