#!/usr/bin/env python3
"""Compare extracted-tree egbeta_limnbr eval_ckpt metrics vs gold (main output/)."""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

SHORT = {"largest_sd": "SD", "largest_gla": "GLA", "largest_gba": "GBA"}


def load_overall(path: str):
    d = json.load(open(path))
    o = d["overall"]
    mae = float(o["MAE"])
    rmse = float(o["RMSE"])
    mape = float(o.get("Masked-MAPE%") or o.get("MAPE%"))
    return mae, rmse, mape, int(d.get("mape_mask_threshold") or 0)


def route_from_path(path: str):
    tagdir = [p for p in path.split("/") if p.startswith("oastid_largest_")][0]
    a = tagdir.split("_to_")
    src = a[0].replace("oastid_", "")
    tgt = a[1].split("_full_")[0]
    # variant from ..._abl_<variant>_eval_ckpt
    m = re.search(r"_abl_([a-z0-9_]+)_eval_ckpt", tagdir)
    variant = m.group(1) if m else "?"
    return f"{SHORT[src]}→{SHORT[tgt]}", variant, tagdir


def index_metrics(root: str):
    files = sorted(
        glob.glob(os.path.join(root, "*egbeta_limnbr_*_eval_ckpt", "*_metrics.json"))
    )
    out = {}
    for f in files:
        route, variant, tagdir = route_from_path(f)
        key = (variant, route)
        out[key] = (*load_overall(f), f, tagdir)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-root", required=True)
    ap.add_argument("--gold-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--atol-mae", type=float, default=1e-4)
    ap.add_argument("--rtol", type=float, default=1e-5)
    # GPU nondeterminism soft tolerance for "essentially same"
    ap.add_argument("--soft-atol", type=float, default=0.05)
    args = ap.parse_args()

    new = index_metrics(args.new_root)
    gold = index_metrics(args.gold_root)

    lines = []
    lines.append(f"new={args.new_root}")
    lines.append(f"gold={args.gold_root}")
    lines.append(f"new_keys={len(new)} gold_keys={len(gold)}")

    keys = sorted(set(new) | set(gold))
    exact = soft = fail = missing = 0
    rows = []
    for k in keys:
        variant, route = k
        if k not in new:
            missing += 1
            rows.append((variant, route, "MISSING_NEW", None))
            continue
        if k not in gold:
            missing += 1
            rows.append((variant, route, "MISSING_GOLD", None))
            continue
        n = new[k]
        g = gold[k]
        diffs = [abs(n[i] - g[i]) for i in range(3)]
        if all(d <= args.atol_mae for d in diffs):
            status = "EXACT"
            exact += 1
        elif all(d <= args.soft_atol for d in diffs):
            status = "SOFT"
            soft += 1
        else:
            status = "DIFF"
            fail += 1
        rows.append(
            (
                variant,
                route,
                status,
                (n[0], n[1], n[2], g[0], g[1], g[2], diffs[0], diffs[1], diffs[2]),
            )
        )

    lines.append(
        f"summary exact={exact} soft(<={args.soft_atol})={soft} "
        f"diff={fail} missing={missing} total={len(keys)}"
    )
    lines.append(
        f"{'variant':28} {'route':8} {'st':6} "
        f"{'newMAE':>8} {'newRMSE':>8} {'newMAPE':>8} "
        f"{'gldMAE':>8} {'gldRMSE':>8} {'gldMAPE':>8} "
        f"{'dMAE':>8} {'dRMSE':>8} {'dMAPE':>8}"
    )
    for variant, route, status, nums in rows:
        if nums is None:
            lines.append(f"{variant:28} {route:8} {status}")
            continue
        nm, nr, np_, gm, gr, gp, dm, dr, dp = nums
        lines.append(
            f"{variant:28} {route:8} {status:6} "
            f"{nm:8.4f} {nr:8.4f} {np_:8.4f} "
            f"{gm:8.4f} {gr:8.4f} {gp:8.4f} "
            f"{dm:8.4f} {dr:8.4f} {dp:8.4f}"
        )

    text = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(text)
    print(text)
    # exit 0 even on soft; nonzero if hard DIFF or missing
    if fail or missing:
        sys.exit(2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
