from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Any, List

import numpy as np

from env.career_env import CareerGuidanceEnv, N_CLUSTERS
from agents.q_learning_agent import _encode_state, QLearningAgent


@dataclass
class EvalResult:
    name: str
    n_episodes: int
    mean_return: float
    mean_questions: float
    accuracy: float
    group_stats: Dict[str, Dict[str, float]]  # group -> {accuracy, questions}


def evaluate_policy(
    env: CareerGuidanceEnv,
    policy_fn: Callable[[np.ndarray], int],
    name: str,
    n_episodes: int = 1000,
) -> EvalResult:
    """Evaluate any policy (agent) in the environment."""
    returns = []
    questions_per_ep = []
    correct_flags = []
    group_records: Dict[str, List[Dict[str, Any]]] = {}

    for _ in range(n_episodes):
        obs, info = env.reset()
        group = info.get("student_group", "unknown")
        done = False
        ep_return = 0.0
        questions = 0
        correct = False

        while not done:
            action = policy_fn(obs)
            next_obs, reward, done, step_info = env.step(action)
            ep_return += reward
            obs = next_obs

            if action < N_CLUSTERS:
                questions += 1

            if done:
                correct = bool(step_info.get("correct", False))

        returns.append(ep_return)
        questions_per_ep.append(questions)
        correct_flags.append(correct)

        if group not in group_records:
            group_records[group] = []
        group_records[group].append(
            {"return": ep_return, "questions": questions, "correct": correct}
        )

    mean_return = float(np.mean(returns))
    mean_questions = float(np.mean(questions_per_ep))
    accuracy = float(np.mean(correct_flags))

    group_stats: Dict[str, Dict[str, float]] = {}
    for group, records in group_records.items():
        arr_ret = np.array([r["return"] for r in records], dtype=float)
        arr_q = np.array([r["questions"] for r in records], dtype=float)
        arr_corr = np.array([r["correct"] for r in records], dtype=float)
        group_stats[group] = {
            "mean_return": float(arr_ret.mean()),
            "mean_questions": float(arr_q.mean()),
            "accuracy": float(arr_corr.mean()),
        }

    return EvalResult(
        name=name,
        n_episodes=n_episodes,
        mean_return=mean_return,
        mean_questions=mean_questions,
        accuracy=accuracy,
        group_stats=group_stats,
    )


def print_eval_result(result: EvalResult) -> None:
    """Pretty-print the evaluation result."""
    print(f"\n=== Evaluation: {result.name} ===")
    print(f"Episodes           : {result.n_episodes}")
    print(f"Mean return        : {result.mean_return:.3f}")
    print(f"Mean # of questions: {result.mean_questions:.3f}")
    print(f"Accuracy           : {result.accuracy:.3f}")

    print("\nPer-group statistics (by student_group):")
    for group, stats in result.group_stats.items():
        print(f"  Group: {group}")
        print(
            f"    Accuracy        : {stats['accuracy']:.3f}\n"
            f"    Mean return     : {stats['mean_return']:.3f}\n"
            f"    Mean questions  : {stats['mean_questions']:.3f}"
        )
