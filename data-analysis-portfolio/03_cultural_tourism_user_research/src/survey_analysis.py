import pandas as pd
from scipy.stats import chi2_contingency


def crosstab_analysis(df: pd.DataFrame, row: str, col: str):
    """Crosstab analysis and chi-square test."""
    table = pd.crosstab(df[row], df[col])
    chi2, p, dof, expected = chi2_contingency(table)
    return table, {"chi2": chi2, "p_value": p, "dof": dof}


def group_summary(df: pd.DataFrame):
    """Summarize key metrics by game-user group."""
    return df.groupby("game_user")[["culture_identity", "nft_awareness", "trust_in_platform", "offline_travel_interest"]].mean()
