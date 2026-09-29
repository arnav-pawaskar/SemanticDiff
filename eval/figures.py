"""Static figures for the report (matplotlib, light theme, reference palette)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE_RAMP = ["#f1f6fd", "#cfe1f7", "#9fc3ef", "#6aa2e6", "#2a78d6", "#1d5aa6", "#123d73"]


def _style(ax, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold", pad=12)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _fig(w, h):
    fig, ax = plt.subplots(figsize=(w, h), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def make_figures(out: Path, taskA, by_cat, danger, test_df, pred_col, cm, cm_labels, perlab) -> None:
    # 1. Task A F1 on test, one bar per system (single series -> no legend, direct labels)
    d = taskA[taskA.split == "test"]
    fig, ax = _fig(8, 4)
    x = np.arange(len(d))
    bars = ax.bar(x, d.f1, width=0.6, color=[SERIES[0] if s.startswith("semanticdiff") else "#9fc3ef"
                                            for s in d.system], edgecolor=SURFACE, linewidth=2)
    for b, v in zip(bars, d.f1):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", color=INK, fontsize=9)
    ax.set_xticks(x, d.system, rotation=20, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("F1 (material change)", color=INK2)
    _style(ax, "Task A — detecting meaning changes (test split)")
    fig.tight_layout()
    fig.savefig(out / "fig_taskA_f1.png", facecolor=SURFACE)
    plt.close(fig)

    # 2. Accuracy per category x system — heatmap, single-hue sequential
    systems = [c for c in by_cat.columns if c not in ("category", "n")]
    M = by_cat[systems].values
    fig, ax = _fig(9, 0.45 * len(by_cat) + 1.6)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(systems)), systems, rotation=25, ha="right")
    ax.set_yticks(range(len(by_cat)), [f"{c} (n={n})" for c, n in zip(by_cat.category, by_cat.n)])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] > 0.6 else INK)
    ax.set_title("Accuracy by change category (test) — paraphrase row = correctly NOT flagged",
                 loc="left", color=INK, fontsize=11, fontweight="bold", pad=10)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig_accuracy_by_category.png", facecolor=SURFACE)
    plt.close(fig)

    # 3. Danger-zone scatter: lexical change vs embedding similarity, coloured by gold
    fig, ax = _fig(7.5, 5.6)
    t = test_df
    for gold, color, label in ((False, SERIES[0], "meaning preserved"), (True, SERIES[1], "meaning changed")):
        g = t[t.gold == gold]
        wrong = g[pred_col["semanticdiff"]].astype(bool) != g.gold
        ax.scatter(g.lexical[~wrong], g.cos[~wrong], s=34, color=color, edgecolor=SURFACE, linewidth=1.5,
                   label=f"{label} (SemanticDiff correct)", zorder=3)
        ax.scatter(g.lexical[wrong], g.cos[wrong], s=48, facecolor="none", edgecolor=color, linewidth=1.8,
                   marker="o", label=f"{label} (SemanticDiff wrong)", zorder=4)
    ax.axvline(0.15, color=INK2, linewidth=1, linestyle=(0, (3, 3)))
    ax.text(0.145, 1.0, "← small edits", color=INK2, fontsize=8, va="bottom", ha="right",
            transform=ax.get_xaxis_transform())
    ax.set_xlabel("Textual change (word-level edit dissimilarity)", color=INK2)
    ax.set_ylabel("Embedding cosine similarity", color=INK2)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2,
              labelcolor=INK2)
    _style(ax, "Textual change ≠ semantic change (test split)")
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(out / "fig_danger_zone.png", facecolor=SURFACE)
    plt.close(fig)

    # 4. Confusion matrix (single-label test pairs)
    fig, ax = _fig(7.5, 6.5)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    ax.imshow(norm, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("blue", BLUE_RAMP), vmin=0, vmax=1)
    ax.set_xticks(range(len(cm_labels)), cm_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(cm_labels)), cm_labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.6 else INK)
    ax.set_xlabel("Predicted", color=INK2)
    ax.set_ylabel("Gold", color=INK2)
    ax.set_title("SemanticDiff change-type confusion (single-label test pairs)", loc="left",
                 color=INK, fontsize=11, fontweight="bold", pad=10)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig_confusion_semanticdiff.png", facecolor=SURFACE)
    plt.close(fig)
