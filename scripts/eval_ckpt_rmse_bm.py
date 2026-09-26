#!/usr/bin/env python
"""Cross-domain: global MAE + batch-mean RMSE (look-0 / mask y!=0)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

SRC = {"sd": "largest_sd", "gla": "largest_gla", "gba": "largest_gba"}
TGTS = {
    "sd": ["largest_gla", "largest_gba"],
    "gla": ["largest_sd", "largest_gba"],
    "gba": ["largest_sd", "largest_gla"],
}


def _install_entry():
    sys.argv = ["eval_rmse_bm", "--no_beta_net", "0", "--skip_stage3", "0"]
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "revin_entry",
        os.path.join(ROOT, "stage2/oastid_full_huber_egbeta_limnbr_revin_entry.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.H


def _bm_one(H, args, model, device, target):
    base = H.base
    pack = base.load_target_eval_pack(args, target, device, model=model)
    ute = pack["ute"]
    cal_start, spd, mps = pack["cal_start"], pack["spd"], pack["mps"]
    test_loader, scaler = pack["test_loader"], pack["scaler"]
    edge_index, edge_weight, node_feat = (
        pack["edge_index"],
        pack["edge_weight"],
        pack["node_feat"],
    )
    sem, behav = pack["sem"], pack["behav"]
    model.eval()
    abs_all, sq_all, n_all = 0.0, 0.0, 0
    mape0_num, mape0_den = 0.0, 0
    mape1_num, mape1_den = 0.0, 0
    mape2_num, mape2_den = 0.0, 0
    bm_u, bm_m = [], []
    t0 = time.perf_counter()
    with torch.no_grad():
        for batch in test_loader:
            x = base.to_dev(batch[0], device)[..., : args.input_dim]
            y = base.to_dev(batch[1], device)[..., : args.output_dim]
            tid, dow = (
                base.time_indices(
                    batch[2],
                    spd,
                    model.time_of_day_size,
                    device,
                    calendar_start=cal_start,
                    minutes_per_step=mps,
                )
                if ute
                else (None, None)
            )
            pred = model(x, edge_index, edge_weight, node_feat, tid, dow, sem, behav)
            yp = scaler.inverse_transform(pred.detach().cpu().numpy())
            yt = scaler.inverse_transform(y.detach().cpu().numpy())
            err = yp - yt
            abs_all += float(np.abs(err).sum())
            sq_all += float((err ** 2).sum())
            n_all += int(err.size)
            bm_u.append(float(np.sqrt(np.mean(err ** 2))))
            mask = np.abs(yt) > 0
            if mask.any():
                bm_m.append(float(np.sqrt(np.mean((err[mask]) ** 2))))
                mape0_num += float(np.abs(err[mask] / yt[mask]).sum())
                mape0_den += int(mask.sum())
            else:
                bm_m.append(float("nan"))
            m1 = np.abs(yt) > 1
            if m1.any():
                mape1_num += float(np.abs(err[m1] / yt[m1]).sum())
                mape1_den += int(m1.sum())
            m2 = np.abs(yt) > 2
            if m2.any():
                mape2_num += float(np.abs(err[m2] / yt[m2]).sum())
                mape2_den += int(m2.sum())
    if device.type == "cuda":
        torch.cuda.synchronize()
    return {
        "target": target,
        "MAE_g": abs_all / max(n_all, 1),
        "RMSE_g": float(np.sqrt(sq_all / max(n_all, 1))),
        "RMSE_bm_look0": float(np.nanmean(bm_u)),
        "RMSE_bm_mask0": float(np.nanmean(bm_m)),
        "MAPE@0": (100.0 * mape0_num / mape0_den) if mape0_den else float("nan"),
        "MAPE@1": (100.0 * mape1_num / mape1_den) if mape1_den else float("nan"),
        "MAPE@2": (100.0 * mape2_num / mape2_den) if mape2_den else float("nan"),
        "n_batch": len(bm_u),
        "infer_seconds": time.perf_counter() - t0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=("sd", "gla", "gba"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", default="eval")
    our = ap.parse_args()
    source = SRC[our.source]
    ckpt_path = our.ckpt if os.path.isabs(our.ckpt) else os.path.join(ROOT, our.ckpt)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(ckpt_path)

    H = _install_entry()
    sys.argv = [
        "eval_rmse_bm",
        "--source",
        source,
        "--targets",
        ",".join(TGTS[our.source]),
        "--eval_ckpt",
        ckpt_path,
        "--device",
        "cuda:0",
        "--mape_mask_threshold",
        "2",
        "--tag",
        f"rmse_bm_{our.name}",
    ]
    cli = H.parse_args()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = H._args_from_ckpt(cli, ckpt["args"]) if hasattr(H, "_args_from_ckpt") else H.base._args_from_ckpt(cli, ckpt["args"])
    args.source = source
    args.targets = ",".join(TGTS[our.source])
    device = torch.device("cuda:0")
    print(f"[bm] {our.name} {source} ckpt={ckpt_path}", flush=True)
    print(
        f"[bm] no_beta_net={getattr(args,'no_beta_net',0)} use_sem={getattr(args,'use_sem_emb',1)} "
        f"use_bsts={getattr(args,'use_bsts',1)}",
        flush=True,
    )
    model = H.build_oastid_model(args).to(device)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    print(f"[bm] load missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    rows = []
    for tgt in TGTS[our.source]:
        row = _bm_one(H, args, model, device, tgt)
        rows.append(row)
        print(
            f"[bm] {source}->{tgt} MAE_g={row['MAE_g']:.4f} RMSE_g={row['RMSE_g']:.4f} "
            f"RMSE_bm_look0={row['RMSE_bm_look0']:.4f} RMSE_bm_mask0={row['RMSE_bm_mask0']:.4f} "
            f"MAPE@0={row['MAPE@0']:.4f} MAPE@1={row['MAPE@1']:.4f} MAPE@2={row['MAPE@2']:.4f} "
            f"nb={row['n_batch']}",
            flush=True,
        )
    out_dir = os.path.join(ROOT, "output", "cross_domain", f"oastid_{source}_rmse_bm_{our.name}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "metrics.json")
    with open(out_path, "w") as f:
        json.dump({"name": our.name, "source": source, "ckpt": ckpt_path, "routes": rows}, f, indent=2)
    print(f"[bm] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
