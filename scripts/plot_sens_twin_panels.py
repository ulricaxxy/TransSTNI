#!/usr/bin/env python
"""Sensitivity panels: left MAE, right RMSE; independent y-lims; one file per axis."""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CD = os.path.join(ROOT, "output", "cross_domain")

for _p in (
    "/usr/share/fonts/liberation/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/liberation/LiberationSerif-Bold.ttf",
    "/usr/share/fonts/urw-base35/NimbusRoman-Regular.otf",
):
    if os.path.isfile(_p):
        fm.fontManager.addfont(_p)

# (tick, stem) — None = sens_default_{src}
AXES: Dict[str, List[Tuple[str, Optional[str]]]] = {
    "gsi": [("4", "gsi4"), ("8", "gsi8"), ("16", "gsi16"), ("32", None)],
    "wti": [
        ("4/4/4/4", "wti4"),
        ("8/8/8/8", "wti8"),
        ("16/16/16/16", "wti16"),
        ("32/32/32/32", "wti32"),
        ("16/8/8/8", None),
    ],
    "mlp": [("1", "mlp1"), ("2", "mlp2"), ("3", None), ("4", "mlp4"), ("5", "mlp5")],
    "sub": [
        ("0.25–0.50", "sub025050"),
        ("0.50–0.75", None),
        ("0.75–1.00", "sub075100"),
        ("0.50–1.00", "sub050100"),
    ],
    "lam": [("0", "lam0"), ("0.5", "lam0.5"), ("1.0", None), ("2.0", "lam2")],
}
DEFAULT_IDX = {"gsi": 3, "wti": 4, "mlp": 2, "sub": 1, "lam": 2}
PANEL = {
    "gsi": r"(a) GSI dim. $d_z$",
    "wti": r"(b) WTI width",
    "mlp": r"(c) Depth $L$",
    "sub": r"(d) Subgraph ratio",
    "lam": r"(e) Stage-2 $\lambda$",
}
FILE_STEM = {
    "gsi": "sens_panel_gsi",
    "wti": "sens_panel_wti",
    "mlp": "sens_panel_mlp",
    "sub": "sens_panel_sub",
    "lam": "sens_panel_lam",
}
ROUTES = [
    ("sd", "largest_gla"),
    ("sd", "largest_gba"),
    ("gla", "largest_sd"),
    ("gla", "largest_gba"),
    ("gba", "largest_sd"),
    ("gba", "largest_gla"),
]

MAE_C = "#1f4e79"
RMSE_C = "#c0392b"
STAR = "#c0392b"
FILL = "#5b8fbf"


def _load(src: str, stem: Optional[str]) -> Optional[dict]:
    name = f"sens_default_{src}" if stem is None else f"sens_{stem}_{src}"
    path = os.path.join(CD, f"oastid_largest_{src}_rmse_bm_{name}", "metrics.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def _metric(obj: dict, tgt: str, key: str) -> Optional[float]:
    for r in obj.get("routes", []):
        if r["target"] == tgt:
            return float(r[key])
    return None


def collect_mean_std(metric: str) -> Dict[str, dict]:
    out = {}
    for axis, pts in AXES.items():
        labels = [p[0] for p in pts]
        means, stds = [], []
        for _lab, stem in pts:
            vals = []
            for src, tgt in ROUTES:
                obj = _load(src, stem)
                if obj is None:
                    continue
                v = _metric(obj, tgt, metric)
                if v is not None:
                    vals.append(v)
            means.append(float(np.mean(vals)) if vals else float("nan"))
            stds.append(float(np.std(vals)) if vals else float("nan"))
        out[axis] = {
            "labels": labels,
            "mean": np.asarray(means, dtype=float),
            "std": np.asarray(stds, dtype=float),
            "default_i": DEFAULT_IDX[axis],
        }
    return out


def _style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Liberation Serif", "Nimbus Roman", "DejaVu Serif", "Times"],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "xtick.minor.size": 0,
            "ytick.minor.size": 0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "axes.spines.top": True,
            "axes.spines.right": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def _ylim(mean: np.ndarray, std: np.ndarray, pad_frac: float = 0.18):
    lo = float(np.nanmin(mean - std))
    hi = float(np.nanmax(mean + std))
    span = max(hi - lo, 1e-3)
    pad = pad_frac * span
    return lo - pad, hi + pad


def plot_one(axis: str, mae: dict, rmse: dict, out_prefix: str) -> None:
    _style()
    d_m, d_r = mae[axis], rmse[axis]
    x = np.arange(len(d_m["labels"]))
    di = d_m["default_i"]

    fig, ax_l = plt.subplots(figsize=(3.6, 2.7))
    ax_r = ax_l.twinx()

    # MAE (left)
    ax_l.fill_between(
        x,
        d_m["mean"] - d_m["std"],
        d_m["mean"] + d_m["std"],
        color=FILL,
        alpha=0.22,
        lw=0,
        zorder=1,
    )
    ax_l.plot(x, d_m["mean"], "-o", color=MAE_C, lw=1.8, ms=5.5, zorder=3, label="MAE")
    ax_l.plot(x[di], d_m["mean"][di], "*", color=STAR, ms=13, zorder=5)

    # RMSE (right)
    ax_r.plot(
        x,
        d_r["mean"],
        "--s",
        color=RMSE_C,
        lw=1.5,
        ms=4.5,
        mfc="white",
        mew=1.2,
        zorder=3,
        label="RMSE",
        alpha=0.95,
    )
    ax_r.plot(x[di], d_r["mean"][di], "*", color=STAR, ms=13, zorder=5)

    ax_l.set_ylim(*_ylim(d_m["mean"], d_m["std"]))
    # RMSE: no shaded band; pad from mean range only
    ax_r.set_ylim(*_ylim(d_r["mean"], np.zeros_like(d_r["mean"]), pad_frac=0.22))

    ax_l.set_xticks(x)
    rot = 18 if axis in ("sub", "wti") else 0
    ax_l.set_xticklabels(d_m["labels"], rotation=rot, ha="right" if rot else "center")
    ax_l.set_xlim(-0.35, len(x) - 0.65)
    ax_l.set_ylabel("MAE")
    ax_r.set_ylabel("RMSE")
    ax_l.yaxis.label.set_color("black")
    ax_r.yaxis.label.set_color("black")
    ax_l.grid(True, axis="y", alpha=0.28, lw=0.6)

    # annotate default MAE
    ax_l.annotate(
        f"{d_m['mean'][di]:.2f}",
        xy=(x[di], d_m["mean"][di]),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        fontsize=8,
        color=MAE_C,
    )

    h1, l1 = ax_l.get_legend_handles_labels()
    h2, l2 = ax_r.get_legend_handles_labels()
    star = plt.Line2D(
        [0],
        [0],
        marker="*",
        color="none",
        markerfacecolor=STAR,
        markeredgecolor=STAR,
        markersize=10,
        label="default",
    )
    ax_l.legend(h1 + h2 + [star], l1 + l2 + ["default"], loc="best", frameon=False, fontsize=8)

    fig.tight_layout()

    # finalize spines / ticks after layout (twinx otherwise resets)
    for ax in (ax_l, ax_r):
        for side in ("top", "bottom", "left", "right"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_color("black")
            ax.spines[side].set_linewidth(0.9)
        ax.tick_params(axis="both", which="both", length=0, width=0)
        plt.setp(ax.get_xticklines(), visible=False, markersize=0)
        plt.setp(ax.get_yticklines(), visible=False, markersize=0)
    ax_l.tick_params(axis="y", which="both", length=0, width=0, labelcolor=MAE_C)
    ax_r.tick_params(axis="y", which="both", length=0, width=0, labelcolor=RMSE_C)
    ax_l.tick_params(axis="x", which="both", length=0, width=0, labelcolor="black")
    ax_r.tick_params(axis="x", which="both", length=0, width=0, labelbottom=False)
    # explicit top edge (twinx can drop the top spine visually)
    ax_l.plot(
        [0, 1],
        [1, 1],
        transform=ax_l.transAxes,
        color="black",
        lw=0.9,
        clip_on=False,
        zorder=20,
        solid_capstyle="projecting",
    )

    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    for ext in ("pdf", "png"):
        path = f"{out_prefix}.{ext}"
        fig.savefig(path)
        print(f"[plot] wrote {path}")
    plt.close(fig)


def main():
    out_dir = os.path.join(ROOT, "output", "figures")
    os.makedirs(out_dir, exist_ok=True)
    mae = collect_mean_std("MAE_g")
    rmse = collect_mean_std("RMSE_bm_mask0")
    for axis in ("gsi", "wti", "mlp", "sub", "lam"):
        print(
            f"[{axis}] MAE={np.round(mae[axis]['mean'], 3)} "
            f"RMSE={np.round(rmse[axis]['mean'], 3)}"
        )
        plot_one(axis, mae, rmse, os.path.join(out_dir, FILE_STEM[axis]))


if __name__ == "__main__":
    main()
