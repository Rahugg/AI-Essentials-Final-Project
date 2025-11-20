from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd

from utils.config import DataConfig


def load_career_dataset(
    config: Optional[DataConfig] = None,
) -> pd.DataFrame:
    """
    Load the main 'career_guidance_english.csv' dataset.

    Parameters
    ----------
    config : DataConfig, optional
        If None, uses the default path in DataConfig.

    Returns
    -------
    pd.DataFrame
    """
    if config is None:
        config = DataConfig()
    df = pd.read_csv(config.csv_path)
    return df


def load_feedback_dataset(path: str = "data/career_guidance_feedback_dataset.csv") -> pd.DataFrame:
    """
    Load the feedback dataset.

    Parameters
    ----------
    path : str
        Path to 'career_guidance_feedback_dataset.csv'.

    Returns
    -------
    pd.DataFrame
    """
    return pd.read_csv(path)


def quick_summaries() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to print a few summaries for exploration.
    Run this module as a script to inspect the data.
    """
    df_main = load_career_dataset()
    df_fb = load_feedback_dataset()

    print("\n=== MAIN DATASET HEAD ===")
    print(df_main.head())

    print("\n=== Career Interests distribution ===")
    print(df_main["Career_Interests"].value_counts())

    print("\n=== Field of Study distribution ===")
    print(df_main["Field_of_Study"].value_counts())

    print("\n=== Career Guidance Satisfaction ===")
    print(df_main["Career_Guidance_Satisfaction"].describe())

    print("\n=== FEEDBACK DATASET HEAD ===")
    print(df_fb.head())

    print("\n=== Improvement Suggestions distribution ===")
    print(df_fb["Improvement_Suggestions"].value_counts())

    return df_main, df_fb


if __name__ == "__main__":
    quick_summaries()
