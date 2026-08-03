"""Turn context_comparison_results.json into the context-hint comparison image.

Reads whatever `test2_context_compare.py` actually produced and renders one row
per configuration, so the image cannot drift from the run it claims to show.

The comparison is deliberately narrow. Every configuration transcribes the
sentence correctly; the only thing that moves is whether the spoken account
number arrives as one identifier or as separate letters. Showing five full
paragraphs would bury that in text nobody reads at article width.

Run it after the comparison script:
    python test2_context_compare.py sample_audio/technical_terms.wav --passes 3
    python plot_context_comparison.py
"""

from __future__ import annotations

import json
import re
import sys

import matplotlib.pyplot as plt

RESULTS_FILE = "context_comparison_results.json"
OUT_FILE = "context_comparison.png"

# Order matters: the three configurations without keywords come first, so the
# rows that never group the identifier read as a block.
ROW_ORDER = [
    ("no_context", "no context"),
    ("prompt_only", "prompt only"),
    ("languages_only", "languages only"),
    ("keywords_only", "keywords only"),
    ("prompt_and_keywords", "prompt + keywords"),
]

INK = "#142430"
MUTED = "#5F7686"
ACCENT = "#199A8E"
BORDER = "#DCE8E6"

# Renderings of the same spoken words, from least to most grouped.
GROUPED = ("AC-42", "AC forty-two")


def extract_identifier(transcript: str) -> str:
    """Pull the few words around the account number out of a full transcript."""
    match = re.search(r"account\s+(.{0,14}?)(?=\.|,| The| the)", transcript)
    return f"account {match.group(1).strip()}" if match else "(not found)"


def pick_pass(passes: list) -> tuple:
    """Choose which pass to show, and say how often it grouped the number.

    Where a configuration grouped the identifier on some passes and not
    others, show the rendering it produced most often and report the count
    alongside. Taking the first grouped pass instead would let one outlier
    represent a configuration that mostly did something else, and hiding the
    count would make an intermittent effect look like a rule.
    """
    grouped = [p for p in passes if any(form in p for form in GROUPED)]
    pool = grouped or passes
    chosen = max(pool, key=lambda p: sum(1 for q in pool if extract_identifier(q) == extract_identifier(p)))
    return chosen, len(grouped), len(passes)


def main() -> None:
    try:
        with open(RESULTS_FILE, encoding="utf-8") as handle:
            results = json.load(handle)
    except FileNotFoundError:
        sys.exit(f"{RESULTS_FILE} not found. Run test2_context_compare.py first.")

    rows = []
    for key, label in ROW_ORDER:
        passes = results.get(key)
        if not passes:
            continue
        transcript, hits, total = pick_pass(passes)
        rows.append({
            "label": label,
            "text": extract_identifier(transcript),
            "grouped": any(form in transcript for form in GROUPED),
            "hits": hits,
            "total": total,
        })

    if not rows:
        sys.exit(f"{RESULTS_FILE} holds no recognised configurations.")

    fig, ax = plt.subplots(figsize=(9, 0.72 * len(rows) + 1.9))
    ax.axis("off")

    ax.text(0.02, 0.94, "How the spoken account number came back",
            fontsize=15, fontweight="bold", color=INK, transform=ax.transAxes)
    ax.text(0.02, 0.855, "Same audio, same delay, one context field changed at a time",
            fontsize=10, color=MUTED, transform=ax.transAxes)

    top = 0.74
    step = 0.74 / len(rows)

    for index, row in enumerate(rows):
        y = top - index * step
        colour = ACCENT if row["grouped"] else MUTED
        weight = "bold" if row["grouped"] else "normal"

        ax.text(0.02, y, row["label"], fontsize=11, color=INK,
                family="monospace", transform=ax.transAxes)
        ax.text(0.36, y, f'"{row["text"]}"', fontsize=12, color=colour,
                fontweight=weight, family="monospace", transform=ax.transAxes)

        if row["total"] > 1:
            ax.text(0.88, y, f'{row["hits"]}/{row["total"]}', fontsize=10,
                    color=colour, transform=ax.transAxes)

        ax.plot([0.02, 0.98], [y - 0.055, y - 0.055], color=BORDER,
                linewidth=0.8, transform=ax.transAxes)

    total_passes = rows[0]["total"]
    footer = (
        "Teal marks a run that grouped the letters into an identifier. "
        f"{'Counts show how many of the ' + str(total_passes) + ' passes did' if total_passes > 1 else 'One pass per configuration'}, "
        "since the effect is intermittent rather than a rule."
    )
    ax.text(0.02, 0.02, footer, fontsize=9, color=MUTED,
            transform=ax.transAxes, wrap=True)

    fig.tight_layout()
    fig.savefig(OUT_FILE, dpi=200, facecolor="white")
    print(f"Saved {OUT_FILE}")

    for row in rows:
        mark = "grouped" if row["grouped"] else "separate letters"
        print(f'  {row["label"]:20s} {row["text"]:28s} {mark} ({row["hits"]}/{row["total"]})')


if __name__ == "__main__":
    main()
