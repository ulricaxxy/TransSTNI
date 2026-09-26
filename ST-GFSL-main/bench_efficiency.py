#!/usr/bin/env python
"""ST-GFSL efficiency: 1-epoch train / 1-pass infer timing on LargeST.

Reports (CUDA synchronized):
  - source_meta_epoch_s : one meta-train step (task_num support/query updates)
  - target_finetune_epoch_s : one full pass over target few-shot loader
  - infer_test_s : one full pass over test loader (no grad)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from copy import deepcopy

import torch
import yaml
from torch_geometric.data import DataLoader

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from datasets import traffic_dataset  # noqa: E402
from maml import STMAML  # noqa: E402
from utils import count_parameters  # noqa: E402


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _peak_mb():
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated() / (1024 ** 2))


def _time_source_epoch(model, source_dataset, task_num, repeats=3, warmup=1):
    times = []
    for i in range(warmup + repeats):
        spt_data, spt_A, qry_data, qry_A = source_dataset.get_maml_task_batch(task_num)
        _sync()
        t0 = time.perf_counter()
        _ = model.meta_train_revise(spt_data, spt_A, qry_data, qry_A)
        _sync()
        dt = time.perf_counter() - t0
        if i >= warmup:
            times.append(dt)
    return float(sum(times) / len(times)), times


def _time_finetune_epoch(model, target_loader, repeats=2, warmup=1):
    maml_model = deepcopy(model.model)
    optimizer = torch.optim.Adam(maml_model.parameters(), lr=model.meta_lr, weight_decay=1e-2)
    times = []
    for i in range(warmup + repeats):
        maml_model.train()
        _sync()
        t0 = time.perf_counter()
        for data, A_wave in target_loader:
            data, A_wave = data.cuda(), A_wave.cuda()
            data.node_num = data.node_num[0]
            out, meta_graph = maml_model(data, A_wave[0].float())
            loss = model.calculate_loss(
                out, data.y, meta_graph, A_wave, "test", loss_lambda=model.loss_lambda
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        _sync()
        dt = time.perf_counter() - t0
        if i >= warmup:
            times.append(dt)
    return float(sum(times) / len(times)), times, maml_model


def _time_infer(maml_model, test_loader, repeats=2, warmup=1):
    times = []
    n_samples = 0
    for i in range(warmup + repeats):
        maml_model.eval()
        n = 0
        _sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            for data, A_wave in test_loader:
                data, A_wave = data.cuda(), A_wave.cuda()
                data.node_num = data.node_num[0]
                _ = maml_model(data, A_wave[0].float())
                n += int(data.x.shape[0])
        _sync()
        dt = time.perf_counter() - t0
        if i >= warmup:
            times.append(dt)
            n_samples = n
    return float(sum(times) / len(times)), times, n_samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dataset", required=True, choices=("largest_sd", "largest_gba", "largest_gla"))
    ap.add_argument("--model", default="GRU")
    ap.add_argument("--target_days", type=int, default=3)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--train_repeats", type=int, default=3)
    ap.add_argument("--infer_repeats", type=int, default=2)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")
    print(
        f"[eff] host cuda={torch.cuda.get_device_name(0)} "
        f"test_dataset={args.test_dataset} model={args.model}",
        flush=True,
    )

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    data_args, task_args, model_args = config["data"], config["task"], config["model"]
    torch.manual_seed(7)

    source_dataset = traffic_dataset(data_args, task_args, "source", test_data=args.test_dataset)
    target_dataset = traffic_dataset(
        data_args, task_args, "target", test_data=args.test_dataset, target_days=args.target_days
    )
    test_dataset = traffic_dataset(data_args, task_args, "test", test_data=args.test_dataset)
    target_loader = DataLoader(
        target_dataset, batch_size=task_args["batch_size"], shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=task_args["test_batch_size"], shuffle=False, num_workers=4, pin_memory=True
    )

    model = STMAML(data_args, task_args, model_args, model=args.model).to(device)
    n_params = int(count_parameters(model))
    print(f"[eff] params={n_params} target_len={len(target_dataset)} test_len={len(test_dataset)}", flush=True)

    torch.cuda.reset_peak_memory_stats()
    src_s, src_runs = _time_source_epoch(
        model, source_dataset, task_args["task_num"], repeats=args.train_repeats, warmup=1
    )
    src_peak = _peak_mb()
    print(f"[eff] source_meta_epoch={src_s:.4f}s runs={[round(x,4) for x in src_runs]} peak={src_peak:.1f}MB", flush=True)

    torch.cuda.reset_peak_memory_stats()
    ft_s, ft_runs, maml_model = _time_finetune_epoch(
        model, target_loader, repeats=max(1, args.train_repeats - 1), warmup=1
    )
    ft_peak = _peak_mb()
    print(f"[eff] target_finetune_epoch={ft_s:.4f}s runs={[round(x,4) for x in ft_runs]} peak={ft_peak:.1f}MB", flush=True)

    torch.cuda.reset_peak_memory_stats()
    inf_s, inf_runs, n_samples = _time_infer(
        maml_model, test_loader, repeats=args.infer_repeats, warmup=1
    )
    inf_peak = _peak_mb()
    per_sample_ms = (inf_s / max(n_samples, 1)) * 1000.0
    print(
        f"[eff] infer_test={inf_s:.4f}s runs={[round(x,4) for x in inf_runs]} "
        f"n={n_samples} per_sample={per_sample_ms:.4f}ms peak={inf_peak:.1f}MB",
        flush=True,
    )

    out = {
        "test_dataset": args.test_dataset,
        "model": args.model,
        "his_num": task_args["his_num"],
        "pred_num": task_args["pred_num"],
        "batch_size": task_args["batch_size"],
        "test_batch_size": task_args["test_batch_size"],
        "task_num": task_args["task_num"],
        "target_days": args.target_days,
        "params": n_params,
        "source_meta_epoch_s": round(src_s, 4),
        "source_meta_epoch_runs": [round(x, 4) for x in src_runs],
        "target_finetune_epoch_s": round(ft_s, 4),
        "target_finetune_epoch_runs": [round(x, 4) for x in ft_runs],
        "infer_test_s": round(inf_s, 4),
        "infer_test_runs": [round(x, 4) for x in inf_runs],
        "infer_n_samples": n_samples,
        "infer_per_sample_ms": round(per_sample_ms, 4),
        "peak_mem_mb": {
            "source_meta": round(src_peak, 1),
            "target_finetune": round(ft_peak, 1),
            "infer": round(inf_peak, 1),
        },
        "cuda": torch.cuda.get_device_name(0),
    }
    out_dir = os.path.join(ROOT, "output", "efficiency", args.test_dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.model}_h{task_args['pred_num']}_eff.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[eff] wrote {out_path}", flush=True)
    print(
        f"[eff] SUMMARY {args.test_dataset}: "
        f"train_source={src_s:.3f}s/epoch  train_finetune={ft_s:.3f}s/epoch  "
        f"infer={inf_s:.3f}s/pass",
        flush=True,
    )


if __name__ == "__main__":
    main()
