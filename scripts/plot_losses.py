"""Plot training loss against step for one or more timestamped MiniMind logs.

Usage: ``uv run python scripts/plot_losses.py results/smoke_loss.png "CPU fp32=results/cpu-smoke.log" ...``
Each series is ``label=path``; the figure is stamped with the logs' date range and the generation time.
"""

import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from parse_log import parse, summary  # noqa: E402


def plot(out: Path, series: dict[str, Path], title: str) -> None:
    """Draw one loss curve per log and save the figure.

    Parameters
    ----------
    out : Path
        PNG to write.
    series : dict of str to Path
        Legend label → timestamped log written by ``cpu_smoke.sh`` or the Kaggle notebook.
    title : str
        Figure title.
    """
    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    stamps = []
    for label, path in series.items():
        points = parse(path.read_text())
        s = summary(points)
        ax.plot([p[1] for p in points], [p[2] for p in points],
                label=f"{label} ({s['it_per_s']:.2f} it/s)")
        stamps += [points[0][0], points[-1][0]]
    day = lambda ts: datetime.fromtimestamp(ts).strftime("%Y-%m-%d")  # noqa: E731
    ax.set(title=title, xlabel="step", ylabel="cross-entropy loss")
    ax.legend()
    fig.text(0.99, 0.01, f"runs {day(min(stamps))} → {day(max(stamps))} · generated "
             f"{datetime.now():%Y-%m-%d %H:%M}", ha="right", fontsize=7, color="grey")
    fig.tight_layout()
    fig.savefig(out, dpi=150)


if __name__ == "__main__":
    target, *pairs = sys.argv[1:]
    plot(Path(target), {k: Path(v) for k, v in (p.split("=", 1) for p in pairs)},
         title="MiniMind 64M, pretraining loss")
