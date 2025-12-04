import pandas as pd
import numpy as np

def preprocess_data(df: pd.DataFrame, is_training: bool = False):
    df = df.copy()

    # ----------------------------------------
    # 1. REMOVE UNUSED COLUMNS
    # ----------------------------------------
    df = df.drop(columns=["player_id", "row_id"], errors="ignore")

    # ----------------------------------------
    # 2. ONE-HOT ENCODING
    # ----------------------------------------
    categorical_features = ["position", "team", "opponent", "game_location"]
    df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

    df = df.astype({col: "int" for col in df.select_dtypes("bool").columns})

    # ----------------------------------------
    # 3. FEATURE ENGINEERING
    # ----------------------------------------
    minute_eps = 1e-6

    df["points_per_min"]     = df["points"]     / (df["minutes_played"] + minute_eps)
    df["rebounds_per_min"]   = df["rebounds"]   / (df["minutes_played"] + minute_eps)
    df["assists_per_min"]    = df["assists"]    / (df["minutes_played"] + minute_eps)
    df["steals_per_min"]     = df["steals"]     / (df["minutes_played"] + minute_eps)
    df["blocks_per_min"]     = df["blocks"]     / (df["minutes_played"] + minute_eps)
    df["turnovers_per_min"]  = df["turnovers"]  / (df["minutes_played"] + minute_eps)
    df["efficiency_per_min"] = df["efficiency"] / (df["minutes_played"] + minute_eps)

    df["usage_rate"] = (df["points"] + df["assists"] + df["turnovers"]) / (df["minutes_played"] + minute_eps)
    df["impact_metric"] = (df["points"] + df["rebounds"] + df["assists"]) / (df["minutes_played"] + minute_eps)
    df["scoring_consistency"] = df["fg_pct"] * df["points_per_min"]

    # ----------------------------------------
    # 4. BINNING FEATURES
    # ----------------------------------------
    df["age_bin"] = pd.cut(df["age"], bins=[0, 24, 28, 32, 100],
                           labels=["young", "prime", "mature", "veteran"])
    df["rest_bin"] = pd.cut(df["rest_days"], bins=[-1, 0, 2, 10],
                            labels=["back_to_back", "normal_rest", "well_rest"])
    df["pm_bin"] = pd.cut(df["plus_minus"], bins=[-100, -1, 1, 100],
                          labels=["negative", "neutral", "positive"])

    df = pd.get_dummies(df, columns=["age_bin", "rest_bin", "pm_bin"], drop_first=True)

    df = df.astype({col: "int" for col in df.select_dtypes("bool").columns})

    # ----------------------------------------
    # 5. FEATURE SELECTION
    # ----------------------------------------
    TOP_FEATURES = [
        'efficiency','points','plus_minus','turnovers','blocks_per_min',
        'turnovers_per_min','team_LAL','opponent_GSW','team_CHA',
        'steals_per_min','team_MIA','steals','points_per_min','opponent_IND',
        'opponent_CHA','opponent_DAL','blocks','position_SF','team_OKC',
        'opponent_DEN','team_DAL','assists_per_min','team_BKN','team_CHI',
        'opponent_LAL','efficiency_per_min','team_HOU','opponent_CLE',
        'opponent_UTA','opponent_LAC','opponent_OKC','scoring_consistency',
        'team_NOP','game_location_Home','age','assists','rebounds_per_min',
        'team_GSW','usage_rate','team_MIN','opponent_MIL','opponent_NOP',
        'age_bin_prime','three_pct','pm_bin_neutral','position_PG','team_PHI',
        'opponent_NYK','opponent_ORL','minutes_played','team_SAC',
        'opponent_DET','rest_days','fg_pct','team_TOR','impact_metric',
        'ft_pct','team_DET','position_PF','team_CLE','rebounds',
        'opponent_SAC','opponent_SAS','team_LAC'
    ]

    required_cols = TOP_FEATURES + (["target"] if is_training else [])

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"❌ Missing expected columns in data: {missing}")

    return df[required_cols].copy()
