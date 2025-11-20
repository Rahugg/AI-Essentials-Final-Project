from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from env.career_env import N_ACTIONS, N_CLUSTERS


@dataclass
class RandomAgent:
    """Baseline: uniform random over all actions."""

    rng: np.random.Generator

    def select_action(self, obs: np.ndarray) -> int:
        return int(self.rng.integers(low=0, high=N_ACTIONS))


@dataclass
class FixedSequenceAgent:
    """
    Baseline: ask once about each cluster in fixed order, then recommend the
    cluster with highest evidence.
    """

    max_questions: int = N_CLUSTERS

    def select_action(self, obs: np.ndarray) -> int:
        evidence = obs[:N_CLUSTERS]
        step = int(obs[-1])  # last element is step index

        if step < self.max_questions:
            cluster_index = step % N_CLUSTERS
            return cluster_index  # question action
        else:
            best_cluster = int(evidence.argmax())
            return N_CLUSTERS + best_cluster  # recommendation action


@dataclass
class GreedyEvidenceAgent:
    """
    Baseline: after a few questions, repeatedly query the most promising
    cluster, then recommend it.
    """

    min_questions: int = 2
    max_questions: int = 6

    def select_action(self, obs: np.ndarray) -> int:
        evidence = obs[:N_CLUSTERS]
        step = int(obs[-1])

        if step < self.min_questions:
            cluster_index = step % N_CLUSTERS
            return cluster_index

        if step < self.max_questions:
            cluster_index = int(evidence.argmax())
            return cluster_index

        best_cluster = int(evidence.argmax())
        return N_CLUSTERS + best_cluster
