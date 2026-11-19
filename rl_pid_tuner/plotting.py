"""Plot helpers (matplotlib is an optional dependency: ``pip install -e .[plot]``)."""
from __future__ import annotations

from pathlib import Path

from .simulation import Response


def plot_step_responses(responses: dict[str, Response], path: str | Path | None = None,
                        title: str = "Closed-loop step response", show_control: bool = True):
    """Overlay several methods' output (and control signal) on one figure.

    ``responses`` maps a label (e.g. ``"RL"``, ``"ZN"``) to a :class:`Response`. If ``path`` is
    given the figure is saved there (PNG) and closed; otherwise the figure is returned.
    """
    import matplotlib

    if path is not None:
        matplotlib.use("Agg", force=True)       # headless-safe when only saving
    import matplotlib.pyplot as plt

    rows = 2 if show_control else 1
    fig, axes = plt.subplots(rows, 1, figsize=(8, 5.5 if show_control else 3.5), sharex=True)
    axes = [axes] if rows == 1 else list(axes)
    sp = None
    for name, r in responses.items():
        if r is None or len(r.t) < 2:
            continue
        sp = r.setpoint
        axes[0].plot(r.t, r.y, label=name)
        if show_control:
            axes[1].plot(r.t, r.u, label=name)
    if sp is not None:
        axes[0].axhline(sp, color="k", ls="--", lw=0.8, label="setpoint")
    axes[0].set_ylabel("output y")
    axes[0].set_title(title)
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="best")
    if show_control:
        axes[1].set_ylabel("control u")
        axes[1].grid(alpha=0.3)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return str(path)
    return fig
