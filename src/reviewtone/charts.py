"""Charts for the research folder."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

INK = "#222222"
MUTED = "#9a9a9a"
NEG = "#d1495b"
POS = "#5b8e7d"
LINE = "#2f5d8a"


def _style(ax, title: str, subtitle: str = "") -> None:
    ax.set_title(title + (f"\n{subtitle}" if subtitle else ""), loc="left", fontsize=11, color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK, labelsize=8)


def _xticks(ax, labels: list[str]) -> None:
    step = max(1, len(labels) // 16)
    ax.set_xticks(range(0, len(labels), step))
    ax.set_xticklabels(labels[::step], rotation=45, ha="right")


def tone_and_counts(q: pd.DataFrame, ticker: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(q))
    colors = [MUTED if s else "#c9c9c9" for s in q["sparse"]]
    ax.bar(x, q["n_reviews"], color=colors)
    for i in np.where(q["sparse"])[0]:
        ax.bar(i, q["n_reviews"].iloc[i], color="none", edgecolor=MUTED, hatch="///")
    ax.set_ylabel("product-opinion reviews", fontsize=9)
    ax2 = ax.twinx()
    ax2.plot(x, q["mean_tone"], color=LINE, marker="o", ms=3, lw=1.6)
    ax2.set_ylabel("mean tone (-1..1)", fontsize=9, color=LINE)
    ax2.spines["top"].set_visible(False)
    _style(ax, f"{ticker}: mean review tone and review counts by fiscal quarter",
           "Hatched bars = sparse quarters (below the minimum review count)")
    _xticks(ax, q["fiscal_quarter"].tolist())
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def negative_swing(q: pd.DataFrame, ticker: str, threshold: float, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4))
    at = q["AbnormalTone"]
    colors = [NEG if v <= threshold else (POS if v > 0 else MUTED) for v in at.fillna(0)]
    alpha = [1.0 if u else 0.35 for u in q["usable"]]
    for i, (v, c, a) in enumerate(zip(at, colors, alpha)):
        if pd.notna(v):
            ax.bar(i, v, color=c, alpha=a)
    ax.axhline(0, color=INK, lw=0.8)
    ax.axhline(threshold, color=NEG, lw=0.8, ls="--")
    ax.text(len(q) - 0.5, threshold, f" call threshold {threshold:+.2f}", color=NEG, fontsize=8, va="top", ha="right")
    _style(ax, f"{ticker}: the negative swing (AbnormalTone)",
           "This quarter's mean tone minus the prior four quarters' mean; faded = not usable for a call")
    _xticks(ax, q["fiscal_quarter"].tolist())
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def tone_vs_sales(panel: pd.DataFrame, ticker: str, holdout_quarters: list[str], path: Path) -> Path:
    p = panel[panel["horizon"] == 0].sort_values("fq_index").dropna(subset=["AbnormalTone", "GrowthSurprise"],
                                                                     how="all")
    fig, ax = plt.subplots(figsize=(10, 4.4))
    x = np.arange(len(p))
    ax.bar(x, p["AbnormalTone"].fillna(0), color=[NEG if c == "slowdown" else MUTED for c in p["call"]],
           label="AbnormalTone (red = slowdown call)")
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_ylabel("AbnormalTone", fontsize=9)
    ax2 = ax.twinx()
    ax2.plot(x, p["GrowthSurprise"] * 100, color=LINE, marker="o", ms=3, lw=1.6, label="Revenue growth surprise")
    ax2.set_ylabel("revenue growth surprise, pp (SEC)", fontsize=9, color=LINE)
    ax2.spines["top"].set_visible(False)
    labels = p["signal_quarter"].tolist()
    held = [i for i, lab in enumerate(labels) if lab in holdout_quarters]
    if held:
        ax.axvspan(min(held) - 0.5, max(held) + 0.5, color="#f2efe6", zorder=0)
        ax.text(min(held) - 0.4, ax.get_ylim()[1], "held-out", fontsize=8, va="top", color=INK)
    miss = p["sales_hit"] == False  # noqa: E712
    ax2.scatter(x[miss.values], (p["GrowthSurprise"] * 100)[miss.values], marker="x", s=60, color=NEG, zorder=5,
                label="miss")
    _style(ax, f"{ticker}: review tone vs. subsequently reported revenue",
           "Growth surprise = YoY revenue growth minus its prior four-quarter average")
    _xticks(ax, labels)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def portfolio_cumulative(q: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4))
    for scheme, color in (("review_count", LINE), ("equal", MUTED)):
        col = f"{scheme}_long_short"
        if col in q:
            ax.plot(q["holding_quarter"], (1 + q[col].fillna(0)).cumprod() - 1, color=color, marker="o", ms=3,
                    label=f"long/short, {scheme.replace('_', '-')} weighted")
    ax.axhline(0, color=INK, lw=0.8)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    _style(ax, "Universe: high-minus-low AbnormalTone portfolio, cumulative return",
           "Formed at each calendar quarter end, held the following quarter")
    _xticks(ax, q["holding_quarter"].tolist())
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
