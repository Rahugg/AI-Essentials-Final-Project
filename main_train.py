from __future__ import annotations

import numpy as np

from env.career_env import CareerGuidanceEnv
from env.student_model import StudentPopulation
from agents.q_learning_agent import QLearningAgent
from utils.config import EnvConfig, RLConfig
from utils.data_loader import load_career_dataset


def main() -> None:
    # Hyperparameters and random seed
    rl_cfg = RLConfig()
    rng = np.random.default_rng(seed=rl_cfg.random_seed)

    # Load real dataset and build data-driven population
    df_main = load_career_dataset()
    population = StudentPopulation(df=df_main, rng=rng)

    env_cfg = EnvConfig()
    env = CareerGuidanceEnv(population=population, env_config=env_cfg, rng=rng)

    # Q-learning agent
    agent = QLearningAgent(cfg=rl_cfg, rng=rng)
    returns = agent.train(env)

    print("Training finished.")
    print(f"Average return over last 1000 episodes: {returns[-1000:].mean():.3f}")
    print(f"Best episode return: {returns.max():.3f}")

    # Save learned Q-table
    agent.save("artifacts/q_table.pkl")
    print("Saved Q-table to artifacts/q_table.pkl")

    # Save training returns for plotting
    np.save("artifacts/returns.npy", returns)
    print("Saved training returns to artifacts/returns.npy")


if __name__ == "__main__":
    main()
