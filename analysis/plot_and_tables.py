from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt


ARTIFACTS_DIR = Path("artifacts")


def load_returns() -> np.ndarray:
    path = ARTIFACTS_DIR / "returns.npy"
    returns = np.load(path)
    return returns


def load_eval_results() -> List[Dict]:
    path = ARTIFACTS_DIR / "eval_results.json"
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


# ----------------- PLOTS -----------------


def plot_learning_curve(window: int = 200) -> None:
    """Plot training return per episode and a moving average."""
    returns = load_returns()
    episodes = np.arange(len(returns))

    # Simple moving average
    if window > 1:
        cumsum = np.cumsum(np.insert(returns, 0, 0))
        moving_avg = (cumsum[window:] - cumsum[:-window]) / float(window)
        ma_x = episodes[window - 1 :]
    else:
        moving_avg = returns
        ma_x = episodes

    plt.figure()
    plt.plot(episodes, returns, alpha=0.3, label="Return per episode")
    plt.plot(ma_x, moving_avg, label=f"{window}-episode moving average")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Q-learning training performance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "learning_curve.png", dpi=300)
    plt.close()
    print("Saved learning curve to artifacts/learning_curve.png")


def plot_overall_policy_comparison() -> None:
    """Bar plots: accuracy and mean questions for each policy."""
    results = load_eval_results()

    names = [r["name"] for r in results]
    accuracies = [r["accuracy"] for r in results]
    questions = [r["mean_questions"] for r in results]

    x = np.arange(len(names))

    # Accuracy plot
    plt.figure()
    plt.bar(x, accuracies)
    plt.xticks(x, names, rotation=20, ha="right")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.0)
    plt.title("Policy comparison: overall accuracy")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "policy_accuracy.png", dpi=300)
    plt.close()
    print("Saved policy accuracy plot to artifacts/policy_accuracy.png")

    # Questions plot
    plt.figure()
    plt.bar(x, questions)
    plt.xticks(x, names, rotation=20, ha="right")
    plt.ylabel("Mean number of questions")
    plt.title("Policy comparison: interaction cost")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "policy_questions.png", dpi=300)
    plt.close()
    print("Saved policy questions plot to artifacts/policy_questions.png")


def plot_fairness_by_group() -> None:
    """
    Plot per-group accuracy for each policy.

    Groups are the 'student_group' labels from evaluation (currently gender).
    """
    results = load_eval_results()

    # Collect all group names
    all_groups = set()
    for r in results:
        for g in r["group_stats"].keys():
            all_groups.add(g)
    groups = sorted(all_groups)

    # Prepare accuracy matrix: shape (n_policies, n_groups)
    acc_matrix = np.zeros((len(results), len(groups)), dtype=float)
    for i, r in enumerate(results):
        for j, g in enumerate(groups):
            stats = r["group_stats"].get(g)
            if stats is not None:
                acc_matrix[i, j] = stats["accuracy"]
            else:
                acc_matrix[i, j] = np.nan

    x = np.arange(len(groups))
    width = 0.8 / len(results)  # total width < 1

    plt.figure()
    for i, r in enumerate(results):
        offset = (i - (len(results) - 1) / 2) * width
        plt.bar(
            x + offset,
            acc_matrix[i],
            width=width,
            label=r["name"],
        )

    plt.xticks(x, groups, rotation=20, ha="right")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.0)
    plt.title("Fairness analysis: accuracy by group (gender)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "fairness_accuracy_by_group.png", dpi=300)
    plt.close()
    print("Saved fairness plot to artifacts/fairness_accuracy_by_group.png")


# ----------------- TABLES (LATEX) -----------------


def latex_overall_table() -> str:
    """
    Produce a LaTeX table summarising overall policy metrics.
    """
    results = load_eval_results()

    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(
        r"\begin{tabular}{lccc}"
        r"\hline"
        r"Policy & Accuracy & Mean Questions & Mean Return \\ \hline"
    )

    for r in results:
        name = r["name"]
        acc = r["accuracy"]
        q = r["mean_questions"]
        ret = r["mean_return"]
        lines.append(
            f"{name} & {acc:.3f} & {q:.2f} & {ret:.3f} \\\\"
        )

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Overall comparison of guidance policies.}")
    lines.append(r"\label{tab:policy_overall}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def latex_fairness_table() -> str:
    """
    Produce a LaTeX table summarising accuracy by group for each policy.
    """
    results = load_eval_results()

    all_groups = set()
    for r in results:
        for g in r["group_stats"].keys():
            all_groups.add(g)
    groups = sorted(all_groups)

    header = "Group"
    for r in results:
        header += f" & {r['name']}"

    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(
        r"\begin{tabular}{" + "l" + "c" * len(results) + "}"
    )
    lines.append(r"\hline")
    lines.append(header + r" \\ \hline")

    for g in groups:
        row = g
        for r in results:
            stats = r["group_stats"].get(g)
            if stats is not None:
                row += f" & {stats['accuracy']:.3f}"
            else:
                row += " & --"
        row += r" \\"
        lines.append(row)

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Accuracy by student group (gender) for each guidance policy.}"
    )
    lines.append(r"\label{tab:fairness_groups}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Generate plots
    plot_learning_curve(window=200)
    plot_overall_policy_comparison()
    plot_fairness_by_group()

    # Generate LaTeX tables and save them as .tex snippets
    overall_tex = latex_overall_table()
    fairness_tex = latex_fairness_table()

    (ARTIFACTS_DIR / "table_overall.tex").write_text(overall_tex, encoding="utf-8")
    print("Wrote LaTeX table to artifacts/table_overall.tex")

    (ARTIFACTS_DIR / "table_fairness.tex").write_text(fairness_tex, encoding="utf-8")
    print("Wrote LaTeX table to artifacts/table_fairness.tex")


if __name__ == "__main__":
    main()
