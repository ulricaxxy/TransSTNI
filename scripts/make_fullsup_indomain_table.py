#!/usr/bin/env python
"""Build STUNet-style in-domain table: H3/H6/H12 + Average for OA-STID fullsup.

Also prints STID / STUNet numbers from STUNet Table 3 for reference.
MAPE column uses MAPE@0 (drop y==0), closest to STUNet's zero-mask MAPE.
"""
from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CD = os.path.join(ROOT, "output", "cross_domain")

# STUNet Table 3 (paper) — Average / H3 / H6 / H12  (MAE, RMSE, MAPE%)
# Sourced from the user-provided table image.
PAPER = {
    "SD": {
        "STID": {
            "H3": (15.25, 25.24, 9.08),
            "H6": (17.32, 29.48, 11.08),
            "H12": (20.81, 36.92, 15.75),
            "Average": (17.86, 31.00, 11.94),
        },
        "STUNet": {
            "H3": (14.87, 23.73, 9.68),
            "H6": (16.82, 27.05, 11.10),
            "H12": (19.92, 32.21, 14.33),
            "Average": (17.32, 27.75, 11.65),
        },
        "PatchSTG": {
            "H3": (14.47, 24.53, 9.60),
            "H6": (16.21, 28.06, 10.74),
            "H12": (18.80, 33.30, 13.16),
            "Average": (16.54, 28.79, 11.12),
        },
    },
    "GBA": {
        "STID": {
            "H3": (17.84, 29.54, 11.98),
            "H6": (20.45, 33.50, 14.80),
            "H12": (24.19, 38.82, 19.12),
            "Average": (20.85, 34.02, 15.32),
        },
        "STUNet": {
            "H3": (16.25, 27.29, 11.96),
            "H6": (18.60, 30.91, 14.46),
            "H12": (21.76, 35.25, 18.41),
            "Average": (18.95, 31.26, 14.90),
        },
        "PatchSTG": {
            "H3": (16.38, 28.40, 11.44),
            "H6": (18.73, 32.11, 13.57),
            "H12": (21.89, 36.70, 17.08),
            "Average": (19.08, 32.52, 14.02),
        },
    },
    "GLA": {
        "STID": {
            "H3": (16.55, 28.03, 9.16),
            "H6": (19.22, 32.74, 11.37),
            "H12": (23.01, 38.60, 16.02),
            "Average": (19.65, 33.40, 12.41),
        },
        "STUNet": {
            "H3": (15.80, 26.51, 9.63),
            "H6": (18.34, 30.78, 12.03),
            "H12": (21.70, 35.98, 16.13),
            "Average": (18.75, 31.26, 12.67),
        },
        "PatchSTG": {
            "H3": (15.38, 27.16, 8.97),
            "H6": (17.63, 31.46, 10.70),
            "H12": (20.66, 36.72, 13.99),
            "Average": (17.98, 32.01, 11.28),
        },
    },
}


def _load(stage: str, key: str):
    src = {"sd": "largest_sd", "gba": "largest_gba", "gla": "largest_gla"}[key]
    path = os.path.join(CD, f"oastid_{src}_fullsup_{stage}", f"{src}_metrics.json")
    if not os.path.isfile(path):
        return None, path
    with open(path) as f:
        obj = json.load(f)
    if "horizons" not in obj or "Average" not in obj:
        return None, path
    return obj, path


def _cell(block, mape_key="MAPE@0"):
    return block["MAE"], block["RMSE"], block[mape_key]


def _fmt(triple):
    mae, rmse, mape = triple
    return f"{mae:.2f}", f"{rmse:.2f}", f"{mape:.2f}"


def main():
    rows = []
    print("======== OA-STID in-domain fullsup (H3/H6/H12 + Average) ========")
    print("MAPE column = MAPE@0 (drop y==0), aligned with STUNet-style zero mask.\n")

    header = (
        "| Dataset | Method | "
        "H3 MAE | H3 RMSE | H3 MAPE | "
        "H6 MAE | H6 RMSE | H6 MAPE | "
        "H12 MAE | H12 RMSE | H12 MAPE | "
        "Avg MAE | Avg RMSE | Avg MAPE |"
    )
    sep = "|" + "|".join(["---"] * 14) + "|"
    lines = [header, sep]

    for ds_key, ds_name in (("sd", "SD"), ("gba", "GBA"), ("gla", "GLA")):
        # paper baselines
        for method in ("STID", "PatchSTG", "STUNet"):
            if method not in PAPER[ds_name]:
                continue
            cells = []
            for h in ("H3", "H6", "H12", "Average"):
                cells.extend(_fmt(PAPER[ds_name][method][h]))
            lines.append(
                f"| {ds_name} | {method} (paper) | " + " | ".join(cells) + " |"
            )

        # our fullsup
        for stage, label in (("s1", "OA-STID (S1 fullsup)"), ("s2", "OA-STID (S2 fullsup)")):
            obj, path = _load(stage, ds_key)
            if obj is None:
                lines.append(f"| {ds_name} | {label} | pending (`{path}`) |")
                print(f"[missing] {stage} {ds_key}: {path}")
                continue
            cells = []
            for h in ("H3", "H6", "H12"):
                cells.extend(_fmt(_cell(obj["horizons"][h])))
            cells.extend(_fmt(_cell(obj["Average"])))
            lines.append(f"| {ds_name} | {label} | " + " | ".join(cells) + " |")
            print(
                f"[{ds_name} {stage}] "
                f"H3={obj['horizons']['H3']['MAE']:.2f}/{obj['horizons']['H3']['RMSE']:.2f}/{obj['horizons']['H3']['MAPE@0']:.2f}  "
                f"H6={obj['horizons']['H6']['MAE']:.2f}/{obj['horizons']['H6']['RMSE']:.2f}/{obj['horizons']['H6']['MAPE@0']:.2f}  "
                f"H12={obj['horizons']['H12']['MAE']:.2f}/{obj['horizons']['H12']['RMSE']:.2f}/{obj['horizons']['H12']['MAPE@0']:.2f}  "
                f"Avg={obj['Average']['MAE']:.2f}/{obj['Average']['RMSE']:.2f}/{obj['Average']['MAPE@0']:.2f}"
            )
            rows.append(obj)

    out_md = os.path.join(ROOT, "output", "figures", "fullsup_indomain_table.md")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w") as f:
        f.write("# In-domain full-supervised (STUNet Table-3 style)\n\n")
        f.write("- Protocol: lag/horizon=12; report steps 3/6/12 + Average.\n")
        f.write("- SD/GBA: Stage1 e=30 clip=1; GLA: Stage1 e=40 clip=5.\n")
        f.write("- MAPE = MAPE@0 (exclude y==0).\n")
        f.write("- STID/PatchSTG/STUNet numbers copied from STUNet Table 3.\n\n")
        f.write("\n".join(lines) + "\n")
    print(f"\n[wrote] {out_md}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
