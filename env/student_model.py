from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


# We use the five interest categories from the dataset:
# 'Healthcare', 'Business', 'Finance', 'Tech', 'Design'
CAREER_CLUSTERS = ("Healthcare", "Business", "Finance", "Tech", "Design")
N_CLUSTERS = len(CAREER_CLUSTERS)


@dataclass
class StudentProfile:
    """
    Latent 'true' profile of a student.

    Attributes
    ----------
    interests : np.ndarray
        Length-5 vector (clusters in CAREER_CLUSTERS) summing to 1.
    gender : str
        Gender label from the dataset.
    location : str
        University_Location from the dataset.
    """
    interests: np.ndarray
    gender: str
    location: str

    def best_cluster_index(self) -> int:
        """Index of maximum interest."""
        return int(self.interests.argmax())

    def best_cluster_name(self) -> str:
        return CAREER_CLUSTERS[self.best_cluster_index()]

    @property
    def group_label(self) -> str:
        """
        Group label used for fairness analysis.
        Here we use gender, but you could change to location or a combination.
        """
        return self.gender
        # Alternative:
        # return f"{self.gender}_{self.location}"


class StudentPopulation:
    """
    Data-driven generator for synthetic students.

    It samples rows from the real dataset and then builds a latent interest
    vector that strongly prefers the row's 'Career_Interests' cluster,
    with some probability mass spread across other clusters.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        rng: np.random.Generator | None = None,
        main_interest_weight: float = 0.7,
    ) -> None:
        """
        Parameters
        ----------
        df : pd.DataFrame
            The main career guidance dataset.
        rng : np.random.Generator, optional
            Random number generator.
        main_interest_weight : float
            Weight assigned to the primary interest cluster (0 < w < 1).
            The remaining mass (1 - w) is spread across other clusters.
        """
        required_cols = {"Career_Interests", "Gender", "University_Location"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Dataset missing required columns: {missing}")

        self.df = df.reset_index(drop=True)
        self.n_rows = len(self.df)
        self.rng = rng if rng is not None else np.random.default_rng(1234)
        self.main_interest_weight = float(main_interest_weight)

        # Map cluster names to indices
        self.cluster_to_index: Dict[str, int] = {
            name: i for i, name in enumerate(CAREER_CLUSTERS)
        }

        # Sanity check: ensure all values fall within our clusters
        unknown_values = set(df["Career_Interests"].unique()) - set(
            CAREER_CLUSTERS
        )
        if unknown_values:
            raise ValueError(
                f"Unexpected career interest values: {unknown_values}. "
                f"Expected subset of {CAREER_CLUSTERS}."
            )

    def sample_student(self) -> StudentProfile:
        """Sample a student profile based on a random row from the dataset."""
        row_idx = int(self.rng.integers(low=0, high=self.n_rows))
        row = self.df.iloc[row_idx]

        main_interest_name = row["Career_Interests"]
        main_idx = self.cluster_to_index[main_interest_name]

        # Build interest vector:
        # main cluster gets 'main_interest_weight',
        # others share the rest equally (small noise for variability).
        base = np.full(N_CLUSTERS, (1.0 - self.main_interest_weight) / (N_CLUSTERS - 1))
        base[main_idx] = self.main_interest_weight

        # Add small Gaussian noise and renormalise
        noise = self.rng.normal(loc=0.0, scale=0.03, size=N_CLUSTERS)
        interests = np.clip(base + noise, 1e-3, None)
        interests = interests / interests.sum()

        profile = StudentProfile(
            interests=interests,
            gender=str(row["Gender"]),
            location=str(row["University_Location"]),
        )
        return profile
