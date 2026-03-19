"""Parse raw NBA game log JSON into GameToken objects and save as Parquet.

Feature mapping from NBA box score stats to the shared GameToken schema:

  Shared (populated):
    goals_for       <- PTS
    goals_against   <- opponent PTS (joined by GAME_ID)
    goal_diff       <- PLUS_MINUS / 2  (approx; PTS - opp_PTS)
    win             <- WL == 'W'
    shots_for       <- FGA
    shots_against   <- opponent FGA
    giveaways       <- TOV
    takeaways       <- STL
    blocks          <- BLK
    pim             <- PF  (personal fouls, rough proxy)
    shooting_pct    <- FG_PCT
    is_home         <- 'vs.' in MATCHUP
    rest_days       <- computed from game dates
    back_to_back    <- rest_days == 1
    game_number     <- cumulative count within season

  NBA-only (stored in extras, zero-filled in feature vector):
    pp_pct, pk_pct, save_pct, corsi_for_pct, fenwick_for_pct,
    xgoals_for, xgoals_against, hits, ot_game, pp_goals_*, pp_opps_*

Usage:
    python -m src.features.nba_tokenizer
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.features.schema import GameToken, FEATURE_NAMES

RAW_DIR = Path(__file__).parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"


def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def _safe_int(val, default: int = 0) -> int:
    try:
        return int(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def parse_nba_season(season_str: str) -> pd.DataFrame:
    """Load NBA game log JSON for a season and return processed DataFrame.

    season_str: e.g. '2015-16'
    """
    raw_path = RAW_DIR / f"nba_games_{season_str}.json"
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw NBA data not found: {raw_path}")

    records = json.loads(raw_path.read_text())
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["TEAM_ID", "GAME_DATE"]).reset_index(drop=True)

    # is_home: MATCHUP "BOS vs. MIL" = home, "BOS @ MIL" = away
    df["is_home"] = df["MATCHUP"].str.contains(r" vs\. ").astype(int)

    # Join opponent stats by GAME_ID
    opp = df[["GAME_ID", "TEAM_ID", "PTS", "FGA", "FG_PCT"]].copy()
    opp.columns = ["GAME_ID", "OPP_TEAM_ID", "OPP_PTS", "OPP_FGA", "OPP_FG_PCT"]
    df = df.merge(
        opp,
        left_on=["GAME_ID"],
        right_on=["GAME_ID"],
    )
    # Drop self-join rows (keep only the opponent's row)
    df = df[df["TEAM_ID"] != df["OPP_TEAM_ID"]].copy()

    # Compute rest days and back-to-back
    df["prev_date"] = df.groupby(["TEAM_ID"])["GAME_DATE"].shift(1)
    df["rest_days"] = (df["GAME_DATE"] - df["prev_date"]).dt.days.fillna(3).clip(upper=7)
    df["back_to_back"] = (df["rest_days"] == 1).astype(int)
    df["game_number"] = df.groupby("TEAM_ID").cumcount() + 1

    # Build tokens
    tokens = []
    for _, row in df.iterrows():
        gf = _safe_int(row.get("PTS"))
        ga = _safe_int(row.get("OPT_PTS") if "OPT_PTS" in row else row.get("OPP_PTS"))
        win = 1 if str(row.get("WL", "")).upper() == "W" else 0

        tok = GameToken(
            sport="nba",
            season=season_str,
            game_id=str(row["GAME_ID"]),
            team_id=str(row["TEAM_ID"]),
            team_abbrev=str(row.get("TEAM_ABBREVIATION", "")),
            opponent_id=str(row.get("OPP_TEAM_ID", "")),
            opponent_abbrev="",
            date=str(row["GAME_DATE"].date()),
            game_number=_safe_int(row.get("game_number")),
            goals_for=gf,
            goals_against=ga,
            win=win,
            ot_game=0,  # NBA doesn't have OT flag in game log; treat as 0
            shots_for=_safe_int(row.get("FGA")),
            shots_against=_safe_int(row.get("OPP_FGA")),
            hits=0,  # no NHL equivalent
            blocks=_safe_int(row.get("BLK")),
            giveaways=_safe_int(row.get("TOV")),
            takeaways=_safe_int(row.get("STL")),
            pim=_safe_int(row.get("PF")),
            # Power play → use free throw stats as rough proxy
            pp_goals_for=_safe_int(row.get("FTM", 0)),
            pp_goals_against=0,
            pp_opps_for=_safe_int(row.get("FTA", 0)),
            pp_opps_against=0,
            # Advanced — zero (no NBA equivalent)
            corsi_for_pct=0.0,
            fenwick_for_pct=0.0,
            xgoals_for=0.0,
            xgoals_against=0.0,
            is_home=_safe_int(row.get("is_home")),
            rest_days=int(row.get("rest_days", 3)),
            back_to_back=_safe_int(row.get("back_to_back")),
            extras={
                "shooting_pct": _safe_float(row.get("FG_PCT")),
                "opp_shooting_pct": _safe_float(row.get("OPP_FG_PCT")),
                "rebounds": _safe_int(row.get("REB")),
                "assists": _safe_int(row.get("AST")),
                "plus_minus": _safe_float(row.get("PLUS_MINUS")),
            },
        )
        tokens.append(tok)

    # Convert to DataFrame using to_vector()
    rows = []
    for tok in tokens:
        row_dict = {
            "sport": tok.sport,
            "season": tok.season,
            "game_id": tok.game_id,
            "team_id": tok.team_id,
            "team_abbrev": tok.team_abbrev,
            "opponent_id": tok.opponent_id,
            "date": tok.date,
            "game_number": tok.game_number,
        }
        vec = tok.to_vector()
        for i, name in enumerate(FEATURE_NAMES):
            row_dict[name] = float(vec[i])
        rows.append(row_dict)

    return pd.DataFrame(rows)


def process_nba_season(season_str: str) -> pd.DataFrame:
    print(f"Processing NBA {season_str}...")
    df = parse_nba_season(season_str)
    if df.empty:
        print("  Empty — skipping")
        return df
    out = PROCESSED_DIR / f"nba_games_{season_str}.parquet"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    n_games = df["game_id"].nunique()
    print(f"  {n_games} games ({len(df)} team-rows) -> {out.name}")
    return df


def process_all_nba_seasons(season_strs: list[str]) -> pd.DataFrame:
    dfs = []
    for s in season_strs:
        try:
            df = process_nba_season(s)
            if not df.empty:
                dfs.append(df)
        except FileNotFoundError as e:
            print(f"  Skipping: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


if __name__ == "__main__":
    from src.fetch.nba_client import nba_season_str
    seasons = [nba_season_str(y) for y in range(2015, 2025)]
    process_all_nba_seasons(seasons)
