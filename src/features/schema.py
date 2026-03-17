"""GameToken dataclass — one token per game from one team's perspective.

Designed to be sport-agnostic at the base level with sport-specific extras.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Ordered list of numeric feature names used to build the feature vector.
# Order matters — downstream code depends on this ordering.
FEATURE_NAMES: list[str] = [
    # Outcome
    "goals_for",
    "goals_against",
    "goal_diff",
    "win",          # 1 = regulation/OT win, 0 = loss
    "ot_game",      # 1 = went to OT/SO
    # Volume
    "shots_for",
    "shots_against",
    "hits",
    "blocks",
    "giveaways",
    "takeaways",
    "pim",          # penalty minutes
    # Efficiency
    "pp_pct",       # power-play %
    "pk_pct",       # penalty-kill %
    "shooting_pct",
    "save_pct",
    "pp_goals_for",
    "pp_goals_against",
    "pp_opps_for",
    "pp_opps_against",
    # Advanced (may be 0/NaN pre-2010 or if unavailable)
    "corsi_for_pct",
    "fenwick_for_pct",
    "xgoals_for",
    "xgoals_against",
    # Context
    "is_home",      # 1 = home, 0 = away
    "rest_days",    # days since last game (capped at 7 for first game / long breaks)
    "back_to_back", # 1 = played yesterday
    "game_number",  # game # in season (1-82)
]

FEATURE_DIM = len(FEATURE_NAMES)
FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}


@dataclass
class GameToken:
    """Single-game token from one team's perspective."""

    sport: str
    season: str           # e.g. "20112012"
    game_id: str
    team_id: str
    team_abbrev: str
    opponent_id: str
    opponent_abbrev: str
    date: str             # ISO date string "YYYY-MM-DD"
    game_number: int      # 1-indexed within season

    # Outcome
    goals_for: int = 0
    goals_against: int = 0
    win: int = 0
    ot_game: int = 0

    # Volume
    shots_for: int = 0
    shots_against: int = 0
    hits: int = 0
    blocks: int = 0
    giveaways: int = 0
    takeaways: int = 0
    pim: int = 0

    # Efficiency
    pp_goals_for: int = 0
    pp_goals_against: int = 0
    pp_opps_for: int = 0
    pp_opps_against: int = 0

    # Advanced
    corsi_for_pct: float = 0.0
    fenwick_for_pct: float = 0.0
    xgoals_for: float = 0.0
    xgoals_against: float = 0.0

    # Context
    is_home: int = 0
    rest_days: int = 3    # default to 3 for first game
    back_to_back: int = 0

    # Sport-specific extras
    extras: dict = field(default_factory=dict)

    @property
    def goal_diff(self) -> int:
        return self.goals_for - self.goals_against

    @property
    def pp_pct(self) -> float:
        return self.pp_goals_for / self.pp_opps_for if self.pp_opps_for > 0 else 0.0

    @property
    def pk_pct(self) -> float:
        saved = self.pp_opps_against - self.pp_goals_against
        return saved / self.pp_opps_against if self.pp_opps_against > 0 else 1.0

    @property
    def shooting_pct(self) -> float:
        return self.goals_for / self.shots_for if self.shots_for > 0 else 0.0

    @property
    def save_pct(self) -> float:
        saves = self.shots_against - self.goals_against
        return saves / self.shots_against if self.shots_against > 0 else 1.0

    def to_vector(self) -> np.ndarray:
        """Return a float32 numpy array matching FEATURE_NAMES order."""
        vec = np.array([
            float(self.goals_for),
            float(self.goals_against),
            float(self.goal_diff),
            float(self.win),
            float(self.ot_game),
            float(self.shots_for),
            float(self.shots_against),
            float(self.hits),
            float(self.blocks),
            float(self.giveaways),
            float(self.takeaways),
            float(self.pim),
            self.pp_pct,
            self.pk_pct,
            self.shooting_pct,
            self.save_pct,
            float(self.pp_goals_for),
            float(self.pp_goals_against),
            float(self.pp_opps_for),
            float(self.pp_opps_against),
            self.corsi_for_pct,
            self.fenwick_for_pct,
            self.xgoals_for,
            self.xgoals_against,
            float(self.is_home),
            float(min(self.rest_days, 7)),
            float(self.back_to_back),
            float(self.game_number),
        ], dtype=np.float32)
        assert len(vec) == FEATURE_DIM, f"Expected {FEATURE_DIM}, got {len(vec)}"
        return vec
