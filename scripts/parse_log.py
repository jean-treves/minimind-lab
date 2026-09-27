"""Parse a timestamped MiniMind training log into (step, loss) points and throughput.

Lines look like ``<unix_ts> Epoch:[1/1](10/300), loss: 8.1234, ...`` (timestamps are added
by ``scripts/cpu_smoke.sh``). Run ``python scripts/parse_log.py results/cpu-smoke.log``.
"""

import re
import sys
from pathlib import Path

LINE = re.compile(r"^(?P<ts>\d+\.\d+) Epoch:\[\d+/\d+\]\((?P<step>\d+)/\d+\), loss: (?P<loss>[\d.]+)")


def parse(text: str) -> list[tuple[float, int, float]]:
    """Return ``(timestamp, step, loss)`` for every logged training line.

    Parameters
    ----------
    text : str
        Content of a log written by ``cpu_smoke.sh``.

    Returns
    -------
    list of tuple
        One ``(ts, step, loss)`` per matching line, in file order.
    """
    return [
        (float(m["ts"]), int(m["step"]), float(m["loss"]))
        for m in map(LINE.match, text.splitlines())
        if m
    ]


def summary(points: list[tuple[float, int, float]]) -> dict[str, float]:
    """First/last loss and steps per second between the first and last logged step.

    Parameters
    ----------
    points : list of tuple
        Output of :func:`parse`, at least two points.

    Returns
    -------
    dict
        ``first_loss``, ``last_loss``, ``steps``, ``it_per_s``.
    """
    (t0, s0, l0), (t1, s1, l1) = points[0], points[-1]
    return {"first_loss": l0, "last_loss": l1, "steps": s1, "it_per_s": (s1 - s0) / (t1 - t0)}


def _demo() -> None:
    log = (
        "100.000 Epoch:[1/1](10/300), loss: 8.5000, logits_loss: 8.5000\n"
        "noise\n"
        "120.000 Epoch:[1/1](30/300), loss: 6.2500, logits_loss: 6.2500\n"
    )
    s = summary(parse(log))
    assert s == {"first_loss": 8.5, "last_loss": 6.25, "steps": 30, "it_per_s": 1.0}, s


if __name__ == "__main__":
    _demo()
    if len(sys.argv) > 1:
        s = summary(parse(Path(sys.argv[1]).read_text()))
        print(
            f"steps {s['steps']}  loss {s['first_loss']:.3f} -> {s['last_loss']:.3f}  "
            f"{s['it_per_s']:.2f} it/s"
        )
