from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, Any

import numpy as np

from env.student_model import StudentPopulation, StudentProfile, CAREER_CLUSTERS, N_CLUSTERS
from utils.config import EnvConfig


# Action encoding:
# 0 .. N_CLUSTERS-1       -> ask about cluster i
# N_CLUSTERS .. 2N_CLUSTERS-1 -> recommend cluster (i = action - N_CLUSTERS)
N_ACTIONS = 2 * N_CLUSTERS


@dataclass
class EnvState:
    """
    Internal environment state.

    Attributes
    ----------
    evidence : np.ndarray
        Integer evidence scores per cluster.
    step_index : int
        Current step in this episode.
    student : StudentProfile
        Underlying 'true' student.
    """
    evidence: np.ndarray
    step_index: int
    student: StudentProfile


class CareerGuidanceEnv:
    """
    Episodic environment for adaptive career guidance.

    The agent asks questions (gathering noisy evidence) and eventually
    recommends one career cluster. Reward:
      - question: negative penalty
      - final recommendation: +1 if correct, else 0
    """

    def __init__(
        self,
        population: StudentPopulation,
        env_config: EnvConfig | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.cfg = env_config or EnvConfig()
        self.population = population
        self.rng = rng if rng is not None else np.random.default_rng(999)

        self._state: EnvState | None = None

    # ------------- Public API (Gym-like) -------------

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Start a new episode and return initial observation + info."""
        student = self.population.sample_student()
        evidence = np.zeros(N_CLUSTERS, dtype=int)
        self._state = EnvState(evidence=evidence, step_index=0, student=student)

        obs = self._observation()
        info = {
            "student_group": student.group_label,
            "gender": student.gender,
            "location": student.location,
        }
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute an action.

        Parameters
        ----------
        action : int
            0..N_CLUSTERS-1 : ask about cluster i
            N_CLUSTERS..2N_CLUSTERS-1 : recommend cluster (i = action - N_CLUSTERS)

        Returns
        -------
        obs : np.ndarray
        reward : float
        done : bool
        info : dict
        """
        if self._state is None:
            raise RuntimeError("Environment must be reset() before step().")

        if not (0 <= action < N_ACTIONS):
            raise ValueError(f"Invalid action {action}; must be 0..{N_ACTIONS-1}")

        state = self._state
        done = False
        info: Dict[str, Any] = {}

        if action < N_CLUSTERS:
            cluster_index = action
            reward = self._handle_question(cluster_index)
        else:
            cluster_index = action - N_CLUSTERS
            reward = self._handle_recommendation(cluster_index)
            done = True
            info["recommended_cluster"] = CAREER_CLUSTERS[cluster_index]
            info["true_best_cluster"] = state.student.best_cluster_name()
            info["correct"] = (
                cluster_index == state.student.best_cluster_index()
            )

        # Increment step counter and enforce max_steps
        state.step_index += 1
        if state.step_index >= self.cfg.max_steps and not done:
            done = True
            info["terminated_due_to_max_steps"] = True

        obs = self._observation()
        return obs, reward, done, info

    # ------------- Internal mechanics -------------

    def _observation(self) -> np.ndarray:
        """Return agent-visible observation: [evidence..., step_index]."""
        assert self._state is not None
        ev = self._state.evidence
        step = self._state.step_index
        return np.concatenate([ev.astype(float), np.array([step], dtype=float)])

    def _handle_question(self, cluster_index: int) -> float:
        """
        Update evidence based on simulated answer.

        Probability of a positive answer is a simple linear function of
        true interest in that cluster.
        """
        assert self._state is not None
        student = self._state.student

        theta = student.interests[cluster_index]  # in [0, 1]
        p_positive = 0.2 + 0.6 * theta  # in [0.2, 0.8]

        positive = self.rng.random() < p_positive
        if positive:
            self._state.evidence[cluster_index] = min(
                self._state.evidence[cluster_index] + 1,
                self.cfg.evidence_max,
            )
        else:
            self._state.evidence[cluster_index] = max(
                self._state.evidence[cluster_index] - 1,
                self.cfg.evidence_min,
            )

        return self.cfg.question_penalty

    def _handle_recommendation(self, cluster_index: int) -> float:
        """Compute terminal reward for recommendation."""
        assert self._state is not None
        best = self._state.student.best_cluster_index()
        if cluster_index == best:
            return self.cfg.correct_recommendation_reward
        return self.cfg.wrong_recommendation_reward
