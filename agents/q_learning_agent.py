from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Tuple
import pickle
from pathlib import Path

import numpy as np

from env.career_env import CareerGuidanceEnv, N_ACTIONS
from utils.config import RLConfig


StateKey = Tuple[int, ...]  # evidence dims + step index


def _encode_state(obs: np.ndarray) -> StateKey:
    """
    Quantise the continuous observation into a discrete state.

    Here we simply round each element to the nearest integer and build a tuple.
    obs structure: [evidence_0, ..., evidence_{K-1}, step_index]
    """
    return tuple(int(round(x)) for x in obs)


@dataclass
class QLearningAgent:
    """Tabular Q-learning agent."""

    cfg: RLConfig
    rng: np.random.Generator

    def __post_init__(self) -> None:
        self.q_table: Dict[StateKey, np.ndarray] = defaultdict(
            lambda: np.zeros(N_ACTIONS, dtype=float)
        )

    # ---------- Persistence ----------

    def save(self, path: str | Path) -> None:
        """Serialise Q-table to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(dict(self.q_table), f)

    def load(self, path: str | Path) -> None:
        """Load Q-table from disk."""
        path = Path(path)
        with path.open("rb") as f:
            loaded: Dict[StateKey, np.ndarray] = pickle.load(f)
        self.q_table = defaultdict(
            lambda: np.zeros(N_ACTIONS, dtype=float), loaded
        )

    # ---------- Policy & learning ----------

    def _epsilon_for_episode(self, episode: int) -> float:
        """Linear decay of epsilon over episodes."""
        if episode >= self.cfg.epsilon_decay_episodes:
            return self.cfg.epsilon_end
        slope = (
            self.cfg.epsilon_end - self.cfg.epsilon_start
        ) / self.cfg.epsilon_decay_episodes
        return self.cfg.epsilon_start + slope * episode

    def choose_action(self, state: StateKey, episode: int) -> int:
        """Epsilon-greedy choice for training."""
        epsilon = self._epsilon_for_episode(episode)
        if self.rng.random() < epsilon:
            return int(self.rng.integers(low=0, high=N_ACTIONS))
        q_values = self.q_table[state]
        return int(np.argmax(q_values))

    def greedy_action(self, state: StateKey) -> int:
        """Greedy action (used at evaluation time)."""
        q_values = self.q_table[state]
        return int(np.argmax(q_values))

    def update(
        self,
        state: StateKey,
        action: int,
        reward: float,
        next_state: StateKey,
        done: bool,
    ) -> None:
        """One Q-learning update."""
        q_values = self.q_table[state]
        q_sa = q_values[action]

        next_q_values = self.q_table[next_state]
        td_target = reward + (
            0.0
            if done
            else self.cfg.gamma * float(np.max(next_q_values))
        )

        q_values[action] = q_sa + self.cfg.alpha * (td_target - q_sa)

    def train(self, env: CareerGuidanceEnv) -> np.ndarray:
        """
        Train over multiple episodes.

        Returns
        -------
        np.ndarray
            Returns (sum of rewards) per episode.
        """
        returns = np.zeros(self.cfg.n_episodes, dtype=float)

        for episode in range(self.cfg.n_episodes):
            obs, info = env.reset()
            state = _encode_state(obs)
            done = False
            ep_return = 0.0

            while not done:
                action = self.choose_action(state, episode)
                next_obs, reward, done, info = env.step(action)
                next_state = _encode_state(next_obs)

                self.update(state, action, reward, next_state, done)
                state = next_state
                ep_return += reward

            returns[episode] = ep_return

        return returns
