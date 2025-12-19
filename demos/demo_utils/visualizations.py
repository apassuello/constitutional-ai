"""
Visualization utilities for Constitutional AI demo.

Creates charts and plots for principle scores, comparisons, and analysis.
"""

from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def create_principle_bar_chart(
    principle_scores: dict[str, float], flagged_principles: list[str]
) -> Image.Image | None:
    """
    Create a bar chart showing principle scores.

    Args:
        principle_scores: Dictionary of principle names to scores
        flagged_principles: List of flagged principle names

    Returns:
        PIL Image of the chart, or None if no data
    """
    if not principle_scores:
        return None

    # Prepare data
    principles = list(principle_scores.keys())
    scores = list(principle_scores.values())
    colors = ["#dc2626" if p in flagged_principles else "#16a34a" for p in principles]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bars
    bars = ax.bar(range(len(principles)), scores, color=colors, alpha=0.7)

    # Customize
    ax.set_xlabel("Constitutional Principle", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title("Constitutional Principle Evaluation", fontsize=14, fontweight="bold")
    ax.set_xticks(range(len(principles)))
    ax.set_xticklabels([p.replace("_", "\n") for p in principles], fontsize=9)
    ax.set_ylim([0, 1.0])
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Threshold")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()

    # Convert to image
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)

    return img


def create_comparison_chart(
    baseline_scores: dict[str, float], constitutional_scores: dict[str, float]
) -> Image.Image | None:
    """
    Create a comparison chart showing before/after principle scores.

    Args:
        baseline_scores: Baseline model scores
        constitutional_scores: Constitutional model scores

    Returns:
        PIL Image of the chart, or None if no data
    """
    if not baseline_scores or not constitutional_scores:
        return None

    # Prepare data
    principles = list(baseline_scores.keys())
    baseline = [baseline_scores[p] for p in principles]
    constitutional = [constitutional_scores[p] for p in principles]

    x = np.arange(len(principles))
    width = 0.35

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create bars
    bars1 = ax.bar(x - width / 2, baseline, width, label="Baseline", color="#ff6b6b", alpha=0.8)
    bars2 = ax.bar(
        x + width / 2, constitutional, width, label="Constitutional", color="#51cf66", alpha=0.8
    )

    # Customize
    ax.set_xlabel("Constitutional Principle", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title("Before vs. After: Principle Scores", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_", "\n") for p in principles], fontsize=9)
    ax.set_ylim([0, 1.0])
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()

    # Convert to image
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)

    return img


def create_improvement_chart(
    baseline_scores: dict[str, float], constitutional_scores: dict[str, float]
) -> Image.Image | None:
    """
    Create a chart showing improvement for each principle.

    Args:
        baseline_scores: Baseline model scores
        constitutional_scores: Constitutional model scores

    Returns:
        PIL Image of the chart, or None if no data
    """
    if not baseline_scores or not constitutional_scores:
        return None

    # Calculate improvements
    principles = list(baseline_scores.keys())
    improvements = [constitutional_scores[p] - baseline_scores[p] for p in principles]
    colors = ["#16a34a" if imp > 0 else "#dc2626" for imp in improvements]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bars
    bars = ax.barh(range(len(principles)), improvements, color=colors, alpha=0.7)

    # Customize
    ax.set_ylabel("Constitutional Principle", fontsize=12, fontweight="bold")
    ax.set_xlabel("Improvement (Score Change)", fontsize=12, fontweight="bold")
    ax.set_title("Constitutional AI Training Impact", fontsize=14, fontweight="bold")
    ax.set_yticks(range(len(principles)))
    ax.set_yticklabels([p.replace("_", " ").title() for p in principles], fontsize=10)
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
    ax.grid(axis="x", alpha=0.3)

    # Add value labels
    for i, (bar, imp) in enumerate(zip(bars, improvements, strict=False)):
        width = bar.get_width()
        label_x = width + (0.02 if width > 0 else -0.02)
        ha = "left" if width > 0 else "right"
        ax.text(label_x, i, f"{imp:+.2f}", ha=ha, va="center", fontsize=9, fontweight="bold")

    plt.tight_layout()

    # Convert to image
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)

    return img


def create_radar_chart(principle_scores: dict[str, float]) -> Image.Image | None:
    """
    Create a radar chart showing principle balance.

    Args:
        principle_scores: Dictionary of principle scores

    Returns:
        PIL Image of the chart, or None if no data
    """
    if not principle_scores or len(principle_scores) < 3:
        return None

    # Prepare data
    principles = list(principle_scores.keys())
    values = list(principle_scores.values())

    # Number of variables
    N = len(principles)

    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    values += values[:1]  # complete the circle
    angles += angles[:1]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))

    # Draw the chart
    ax.plot(angles, values, "o-", linewidth=2, color="#3b82f6", alpha=0.8)
    ax.fill(angles, values, alpha=0.25, color="#3b82f6")

    # Customize
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([p.replace("_", "\n").title() for p in principles], fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
    ax.grid(True)
    ax.set_title("Principle Balance", fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()

    # Convert to image
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)

    return img
