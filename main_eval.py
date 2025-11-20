from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from env.career_env import CareerGuidanceEnv
from env.student_model import StudentPopulation
from agents.q_learning_agent import QLearningAgent, _encode_state
from agents.baseline_agents import (
    RandomAgent,
    FixedSequenceAgent,
    GreedyEvidenceAgent,
)
from utils.config import EnvConfig, RLConfig
from utils.data_loader import load_career_dataset
from utils.evaluation import evaluate_policy, print_eval_result, EvalResult


def result_to_dict(res: EvalResult) -> dict:
    """Convert EvalResult to a JSON-serialisable dict."""
    return {
        "name": res.name,
        "n_episodes": res.n_episodes,
        "mean_return": res.mean_return,
        "mean_questions": res.mean_questions,
        "accuracy": res.accuracy,
        "group_stats": res.group_stats,
    }


def main() -> None:
    rl_cfg = RLConfig()
    rng = np.random.default_rng(seed=rl_cfg.random_seed)

    df_main = load_career_dataset()
    population = StudentPopulation(df=df_main, rng=rng)

    env_cfg = EnvConfig()
    env = CareerGuidanceEnv(population=population, env_config=env_cfg, rng=rng)

    # Q-learning agent (load trained table)
    q_agent = QLearningAgent(cfg=rl_cfg, rng=rng)
    q_agent.load("artifacts/q_table.pkl")

    def q_policy(obs):
        state = _encode_state(obs)
        return q_agent.greedy_action(state)

    # Baselines
    rand_agent = RandomAgent(rng=rng)
    fixed_agent = FixedSequenceAgent()
    greedy_agent = GreedyEvidenceAgent()

    n_eval_episodes = 2000

    res_q = evaluate_policy(env, q_policy, "Q-learning (greedy)", n_eval_episodes)
    res_rand = evaluate_policy(
        env, rand_agent.select_action, "Random policy", n_eval_episodes
    )
    res_fixed = evaluate_policy(
        env, fixed_agent.select_action, "Fixed-sequence heuristic", n_eval_episodes
    )
    res_greedy = evaluate_policy(
        env, greedy_agent.select_action, "Greedy-evidence heuristic", n_eval_episodes
    )

    for res in (res_q, res_rand, res_fixed, res_greedy):
        print_eval_result(res)

    # Save to JSON for plotting / tables
    metrics = [result_to_dict(r) for r in (res_q, res_rand, res_fixed, res_greedy)]
    out_path = Path("artifacts/eval_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved evaluation metrics to {out_path}")


if __name__ == "__main__":
    main()
