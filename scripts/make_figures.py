"""Figures for REPORT.md from results/ (numbers only; no CT pixels are drawn).

Usage: python scripts/make_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "figures"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
METRICS = (("precision", "Crossing precision"), ("recall", "Crossing recall"),
           ("count_error", "Winding count error"), ("phase_mae", "Phase MAE (windings)"))

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 9, "axes.titlesize": 9.5, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
})


def load_compare(path):
    return json.loads(path.read_text()) if path.exists() else None


def dot_rows(axes, rows, groups, colors, offsets):
    """rows: list of (label, run entry); one dot + 95 % interval per group per row."""
    for ax, (key, title) in zip(axes, METRICS):
        for gi, (group, color, off) in enumerate(zip(groups, colors, offsets)):
            for ri, (_, entry) in enumerate(rows):
                info = entry["groups"].get(group)
                if not info:
                    continue
                y = ri + off
                lo, hi = info["ci95"][key]
                ax.plot([lo, hi], [y, y], color=color, lw=2, solid_capstyle="round")
                ax.plot(info[key], y, "o", ms=6, color=color, mec=SURFACE, mew=1.5)
        ax.set_title(title, loc="left")
        ax.set_yticks(range(len(rows)))
        ax.grid(axis="y", visible=False)
    axes[0].invert_yaxis()  # shared y: invert once, first row on top
    axes[0].set_yticklabels([label for label, _ in rows])
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)


def fig_reconstruction():
    data = load_compare(RES / "compare_pherc0139_scans.json")
    if not data:
        return
    rows = [(e["label"], e) for e in data["runs"]]
    groups = [g for g in rows[0][1]["groups"] if g != "all"]
    fig, axes = plt.subplots(1, 4, figsize=(11, 0.55 * len(rows) + 1.6), sharey=True)
    dot_rows(axes, rows, groups, [BLUE, ORANGE], [-0.14, 0.14])
    handles = [plt.Line2D([], [], color=c, marker="o", lw=2, label=f"slabs {g}") for g, c in zip(groups, [BLUE, ORANGE])]
    fig.legend(handles=handles, loc="upper left", ncol=2, bbox_to_anchor=(0.0, 1.0))
    fig.suptitle("PHerc0139, same 256 rays and labels: only the CT changes (95 % slab-bootstrap intervals)",
                 x=0.0, ha="left", y=1.07, fontsize=10.5, fontweight="bold")
    fig.savefig(FIG / "fig1_scan_and_reconstruction.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_label_pairs():
    spacing = json.loads((RES / "winding_spacing_pherc0139_l2_z3904.json").read_text())
    run = RES / "runs" / "gate_pherc0139_2399_l2_z3904_rerun" / "result.json"
    agreement = json.loads(run.read_text())["label_agreement_by_winding"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True, gridspec_kw={"hspace": 0.35})
    flagged = {(41, 42), (45, 46), (46, 47)}
    for p in spacing["pairs"]:
        x = p["from"] + 0.5
        color = ORANGE if (p["from"], p["to"]) in flagged else BLUE
        a1.plot([x, x], [p["p10"], min(p["p90"], 60)], color=color, lw=2, solid_capstyle="round")
        a1.plot(x, p["median"], "o", color=color, ms=5, mec=SURFACE, mew=1.2)
    a1.set_ylim(-2, 62)
    a1.set_ylabel("gap to next winding\n(level-2 voxels, along normal)")
    a1.set_title("Distance from each labelled winding mesh to the next one (median, 10th to 90th percentile)", loc="left")
    notes = (((41.5, 1), (38.2, 47), "w042 sits on w041"),
             ((45.5, 0.5), (44.0, 56), "w046 sits on w045"),
             ((46.5, 31), (50.5, 50), "w046 to w047: about two gaps"))
    for xy, text_xy, text in notes:
        a1.annotate(text, xy, xytext=text_xy, ha="center", fontsize=8, color=INK2,
                    arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.6})
    a1.text(1.0, -0.12, "upper whiskers clipped at 60 where a mesh pair overlaps only partly",
            transform=a1.transAxes, ha="right", fontsize=7, color=INK2)
    ws = sorted(int(w) for w in agreement)
    vals = [agreement[str(w)]["frac_within_tolerance"] for w in ws]
    colors = [ORANGE if 40 <= w <= 49 else BLUE for w in ws]
    a2.bar(ws, vals, width=0.6, color=colors)
    a2.set_ylim(0, 0.8)
    a2.set_ylabel("labels with a predicted\ncrossing within 2 samples")
    a2.set_title("Agreement between the model and each winding's labels (gate run, 256 slabs)", loc="left")
    a2.set_xticks(range(ws[0], ws[-1] + 1, 2))
    a2.set_xticklabels([f"w{w:03d}" for w in range(ws[0], ws[-1] + 1, 2)], rotation=45)
    a2.axvspan(39.5, 49.5, color=ORANGE, alpha=0.08, lw=0)
    fig.savefig(FIG / "fig3_label_pairs_pherc0139.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_example_slab():
    path = RES / "example_slabs_w041_w043.json"
    if not path.exists():
        return
    ex = json.loads(path.read_text())["examples"][2]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    cols = ex["centre_row_columns"]
    for j, c in enumerate(cols):
        ax.plot(c["labels_t"], [j] * len(c["labels_t"]), "o", ms=2.6, color=BLUE, mew=0)
        ax.plot(c["passages_t"], [j] * len(c["passages_t"]), "o", ms=1.6, color=ORANGE, mew=0)
    mid = cols[len(cols) // 2]
    for t, w in zip(mid["labels_t"], mid["labels_w"]):
        ax.annotate(f"w{w:03d}", (t, len(cols) + 2), ha="center", fontsize=7, color=INK2, rotation=90,
                    annotation_clip=False)
    ax.set_xlim(0, 384)
    ax.set_ylim(-2, len(cols) + 1)
    ax.set_xlabel("position along the ray (samples of about 9.6 um)")
    ax.set_ylabel("column across the slab")
    handles = [plt.Line2D([], [], color=BLUE, marker="o", ls="", label="labelled crossing (published meshes)"),
               plt.Line2D([], [], color=ORANGE, marker="o", ls="", label="model phase passage")]
    ax.legend(handles=handles, loc="lower left")
    ax.grid(axis="y", visible=False)
    ax.set_title("One PHerc0139 slab through windings w043 to w055: labels and predictions, centre row of columns",
                 loc="left", y=1.14)
    fig.savefig(FIG / "fig4_example_slab_pherc0139.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_unseen():
    data = load_compare(RES / "compare_scrolls.json")
    if not data:
        return
    rows = [(e["label"], e) for e in data["runs"]]
    fig, axes = plt.subplots(1, 4, figsize=(11, 0.5 * len(rows) + 1.3), sharey=True)
    dot_rows(axes, rows, ["all"], [BLUE], [0.0])
    fig.suptitle("Training scrolls against unseen scrolls, 256 slabs each (95 % slab-bootstrap intervals)",
                 x=0.0, ha="left", y=1.04, fontsize=10.5, fontweight="bold")
    fig.savefig(FIG / "fig2_scrolls.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    fig_reconstruction()
    fig_label_pairs()
    fig_example_slab()
    fig_unseen()
    print("figures:", sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
