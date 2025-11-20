from dataclasses import dataclass

@dataclass(frozen=True)
class DataConfig:
    """Configuration related to the external dataset."""
    csv_path: str = "data/career_guidance_english.csv"


@dataclass(frozen=True)
class EnvConfig:
    """Configuration for the RL environment."""
    max_steps: int = 10
    question_penalty: float = -0.05
    correct_recommendation_reward: float = 1.0
    wrong_recommendation_reward: float = 0.0
    evidence_min: int = -3
    evidence_max: int = 3


@dataclass(frozen=True)
class RLConfig:
    """Hyperparameters for tabular Q-learning."""
    n_episodes: int = 10_000
    gamma: float = 0.95
    alpha: float = 0.1              # learning rate
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 8_000
    random_seed: int = 42
