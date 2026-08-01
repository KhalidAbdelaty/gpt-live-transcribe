"""Turn delay_benchmark_results.json (from test3_delay_benchmark.py) into the
delay-versus-latency chart used in the "Benchmarking the Five Delay Levels"
section.

Run with:
    python plot_delay_benchmark.py
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt

DELAY_ORDER = ["minimal", "low", "medium", "high", "xhigh"]


def load_summary(path: str = "delay_benchmark_results.json") -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {row["delay"]: row for row in data["summary"]}


def plot(summary: dict, out_path: str = "delay_benchmark_chart.png") -> None:
    labels = [d for d in DELAY_ORDER if d in summary]
    first_delta = [summary[d]["median_time_to_first_delta_s"] for d in labels]
    final = [summary[d]["median_time_to_final_s"] for d in labels]

    # Two panels rather than one grouped chart. Time to final is dominated by
    # when the client commits, so plotting both on one axis buries the
    # first-delta differences under bars that are five times taller and flat.
    fig, (ax_first, ax_final) = plt.subplots(1, 2, figsize=(11, 4.5))
    x = range(len(labels))

    ax_first.bar(x, first_delta, color="#1f77b4")
    for i, value in zip(x, first_delta):
        ax_first.text(i, value, f"{value:.2f}s", ha="center", va="bottom", fontsize=9)
    ax_first.set_title("Time to first partial transcript")
    ax_first.set_ylabel("Seconds (median)")
    ax_first.set_ylim(0, max(first_delta) * 1.25)

    ax_final.bar(x, final, color="#ff7f0e")
    for i, value in zip(x, final):
        ax_final.text(i, value, f"{value:.1f}s", ha="center", va="bottom", fontsize=9)
    ax_final.set_title("Time to final transcript (set by commit timing)")
    ax_final.set_ylabel("Seconds (median)")
    ax_final.set_ylim(0, max(final) * 1.25)

    for ax in (ax_first, ax_final):
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.set_xlabel("delay setting")
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("gpt-live-transcribe delay levels, measured on this audio and connection")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    plot(load_summary())
