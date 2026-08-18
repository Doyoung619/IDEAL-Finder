from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATUS_COLORS = {
    "completed": "#36c98f",
    "in_progress": "#f2b84b",
    "pending": "#334155",
    "blocked": "#ef6461",
}
STATUS_VALUES = {
    "completed": 1.0,
    "in_progress": 0.5,
    "pending": 0.0,
    "blocked": 0.0,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render implementation progress")
    parser.add_argument(
        "--status",
        type=Path,
        default=PROJECT_ROOT / "implementation_progress.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "progress.png",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    state = json.loads(args.status.read_text(encoding="utf-8"))
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    args.status.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    phases = state["phases"]
    names = [phase["name"] for phase in phases]
    statuses = [phase["status"] for phase in phases]
    values = [STATUS_VALUES[status] for status in statuses]
    colors = [STATUS_COLORS[status] for status in statuses]
    completed = statuses.count("completed")
    tests = state["tests"]

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    figure, axis = plt.subplots(figsize=(12, 7), facecolor="#0b1220")
    axis.set_facecolor("#0b1220")
    bars = axis.barh(names[::-1], values[::-1], color=colors[::-1], height=0.58)
    axis.set_xlim(0, 1)
    axis.set_xticks([])
    axis.tick_params(axis="y", colors="#e5edf8", labelsize=11)
    for spine in axis.spines.values():
        spine.set_visible(False)
    for bar, phase in zip(bars, phases[::-1]):
        status = phase["status"].replace("_", " ").upper()
        axis.text(
            0.03,
            bar.get_y() + bar.get_height() / 2,
            f"{status}  |  {phase['detail']}",
            va="center",
            ha="left",
            color="#f8fafc",
            fontsize=9,
            fontweight="bold",
        )

    figure.suptitle(
        state["title"],
        color="#f8fafc",
        fontsize=22,
        fontweight="bold",
        y=0.96,
    )
    axis.set_title(
        (
            f"Branch {state['branch']} @ {state['baseline_commit']}   •   "
            f"Completed phases {completed}/{len(phases)}   •   "
            f"Tests {tests['current_passed']} passed / "
            f"{tests['current_failed']} failed\n"
            f"Updated {state['updated_at']}"
        ),
        color="#9fb1c8",
        fontsize=10,
        pad=22,
    )
    figure.text(
        0.5,
        0.035,
        "Status bars describe verified phase state; they are not synthetic ETA estimates.",
        ha="center",
        color="#71839a",
        fontsize=9,
    )
    figure.subplots_adjust(left=0.19, right=0.97, top=0.78, bottom=0.12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150, facecolor=figure.get_facecolor())
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
