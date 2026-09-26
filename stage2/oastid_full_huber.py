"""OA-STID Huber-loss pipeline. Does NOT modify ``stage2/oastid_full.py``.

Training loss only (val / Cross stay MAE so numbers stay comparable):

  * Stage1: Huber on the **original** traffic scale (same space as
    ``masked_mae_real`` / inverse-std), default ``delta=23``.
  * Stage2 ERM / MAML / MLDG: Huber on the **normalized** scale (same space as
    ``torch.mean(|pred-y|)``), default ``delta=0.15``. Inner-loop support and
    outer query / MLDG terms all use this Stage2 delta.

Huber here is the MAE-consistent piecewise form (PyTorch SmoothL1 / SmoothL1
with ``beta=delta``), written out in full — not a one-line wrapper, not a
collapsed Linear, not textbook-Huber with linear slope ``±delta`` (that would
scale Stage1 grads by 23 and invalidate the current lr).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import OrderedDict

import numpy as np
import torch
from torch.func import functional_call

from lib.dataloader import get_dataloader
from lib.ydzt_sampler_random import build_sizes

from stage2 import oastid_full as base


# Bind every helper the copied Stage1/Stage2 loops need, so those loops stay
# structurally identical to ``oastid_full.py`` (only the train-loss lines change).
PROJECT_ROOT = base.PROJECT_ROOT
make_loader_args = base.make_loader_args
calendar_of = base.calendar_of
load_graph = base.load_graph
load_sem_emb = base.load_sem_emb
time_indices = base.time_indices
to_dev = base.to_dev
masked_mae_real = base.masked_mae_real
sample_task = base.sample_task
sample_subgraph_batch = base.sample_subgraph_batch
sample_mldg_domain_pair = base.sample_mldg_domain_pair
build_graph = base.build_graph
_compute_lr = base._compute_lr
_stage2_permute_adj_flag = base._stage2_permute_adj_flag
_stage2_phase_exchange_flag = base._stage2_phase_exchange_flag
_stage2_period_len = base._stage2_period_len
_stage2_phase_exchange_mode = base._stage2_phase_exchange_mode
_stage2_phase_week_align_flag = base._stage2_phase_week_align_flag
_stage2_phase_preprocess_flag = base._stage2_phase_preprocess_flag
maybe_run_stage2_cross = base.maybe_run_stage2_cross
collect_source_arrays = base.collect_source_arrays
build_oastid_model = base.build_oastid_model
_resolve_sem_flag = base._resolve_sem_flag
eval_target = base.eval_target
parse_target_list = base.parse_target_list
# Re-export base APIs patches expect on the Huber module (same as compiled pipeline).
load_behav_pi = base.load_behav_pi
build_behav_meta_context = base.build_behav_meta_context
save_stage1_ckpt = getattr(base, 'save_stage1_ckpt', None)
_ckpt_dir_for = getattr(base, '_ckpt_dir_for', None)
load_compatible_state_dict = getattr(base, 'load_compatible_state_dict', None)


# ---------------------------------------------------------------------------
# Huber (full piecewise; MAE-matched linear slope)
# ---------------------------------------------------------------------------
def huber_mean(pred, true, delta):
    """Elementwise Huber, then mean over all remaining elements.

    Residual ``e = pred - true``. For ``δ = delta > 0``:

        L(e) = 0.5 * e² / δ          if |e| ≤ δ     (quadratic / L2-like)
             = |e| - 0.5 * δ         if |e| >  δ     (linear, slope ±1 like MAE)

    Continuity: at |e|=δ both sides equal 0.5 δ.
    Derivative: at |e|=δ both sides equal sign(e)  (matches MAE outside).

    This is NOT the textbook form ``0.5 e²`` / ``δ(|e|-0.5δ)``, whose linear
    slope is ``±δ``. With Stage1 δ=23 that form would multiply outlier grads
    by 23 vs MAE and require a new learning rate. The form above is the
    standard MAE → Huber drop-in (PyTorch ``smooth_l1_loss(..., beta=δ)``).
    """
    if pred.shape != true.shape:
        raise ValueError(
            f'huber_mean shape mismatch: pred={tuple(pred.shape)} true={tuple(true.shape)}')
    delta = float(delta)
    if not np.isfinite(delta) or delta <= 0.0:
        raise ValueError(f'Huber delta must be a positive finite scalar, got {delta}')
    err = pred - true
    abs_err = err.abs()
    quadratic = 0.5 * err.square() / delta
    linear = abs_err - 0.5 * delta
    per_elem = torch.where(abs_err <= delta, quadratic, linear)
    return per_elem.mean()


def masked_huber_real(pred, y, scaler, mask_value, delta):
    """Stage1 train Huber on the original (inverse-std) scale.

    Same mask as ``MAE_torch`` / ``masked_mae_real``: keep entries with
    ``true > mask_value`` (drop near-zero / missing sensors). Inverse-transform
    first so ``delta=23`` is in vehicles / 15 min, not z-score units.
    """
    p = scaler.inverse_transform(pred)
    t = scaler.inverse_transform(y)
    if mask_value is not None:
        mask = torch.gt(t, mask_value)
        p = torch.masked_select(p, mask)
        t = torch.masked_select(t, mask)
    return huber_mean(p, t, delta)


def huber_norm(pred, y, delta):
    """Stage2 train Huber on the standardized (z-score) scale."""
    return huber_mean(pred, y, delta)


def _huber_delta_stage1(args):
    return float(getattr(args, 'huber_delta_stage1', 23.0))


def _huber_delta_stage2(args):
    return float(getattr(args, 'huber_delta_stage2', 0.15))


# =============================== Stage 1 ================================== #
def train_source(args, model, device):
    """Stage1 Huber pretrain on the full source graph (not simplified).

    Faithful to the compiled Huber pipeline / ``sage_nosubgraph_patch`` copy:
    loaders / graph / sem / behav / Adam+MultiStepLR / Huber train / MAE val /
    early stop / restore best. Passes ``behav`` into ``model`` (None when
    ``use_behav_emb=0``).
    """
    src_args = make_loader_args(args, args.source)
    ute = args.use_time_emb
    cal_start, spd, mps = calendar_of(args.source, getattr(args, 'calendar_start', None))
    train_loader, val_loader, _, scaler = get_dataloader(
        src_args, normalizer='std', tod=ute, dow=ute, weather=False,
        single=False, return_index=ute)
    edge_index, edge_weight, node_feat = load_graph(args, args.source, device)
    sem = load_sem_emb(args, args.source, device)
    behav = load_behav_pi(args, args.source, device)
    d1 = _huber_delta_stage1(args)

    def tds(batch):
        if not ute:
            return None, None
        return time_indices(
            batch[2], spd, model.time_of_day_size, device,
            calendar_start=cal_start, minutes_per_step=mps)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[int(args.stage1_epochs * 0.5), int(args.stage1_epochs * 0.75)], gamma=0.5)

    print(f'[stage1/huber] train Huber on original scale delta={d1} '
          f'(val stays MAE / masked_mae_real)', flush=True)

    best_val, best_state, bad = float('inf'), None, 0
    for epoch in range(1, args.stage1_epochs + 1):
        model.train()
        t0, tr_loss, nb = time.perf_counter(), 0.0, 0
        for batch in train_loader:
            x = to_dev(batch[0], device)[..., :args.input_dim]
            y = to_dev(batch[1], device)[..., :args.output_dim]
            tid, dow = tds(batch)
            opt.zero_grad()
            loss = masked_huber_real(
                model(x, edge_index, edge_weight, node_feat, tid, dow, sem, behav),
                y, scaler, args.mask_value, d1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            tr_loss += loss.item()
            nb += 1
        sched.step()
        model.eval()
        vl, vb = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                x = to_dev(batch[0], device)[..., :args.input_dim]
                y = to_dev(batch[1], device)[..., :args.output_dim]
                tid, dow = tds(batch)
                vl += masked_mae_real(
                    model(x, edge_index, edge_weight, node_feat, tid, dow, sem, behav),
                    y, scaler, args.mask_value).item()
                vb += 1
        val_mae = vl / max(vb, 1)
        print(f'[stage1][{args.source}] epoch {epoch:03d} train_huber={tr_loss/max(nb,1):.4f} '
              f'val_mae={val_mae:.4f} ({time.perf_counter()-t0:.1f}s)', flush=True)
        save_every = int(getattr(args, 'save_every', 0) or 0)
        save_from = int(getattr(args, 'save_from', 40) or 0)
        if save_every > 0 and epoch >= save_from and epoch % save_every == 0:
            snap_dir = _ckpt_dir_for(args) if _ckpt_dir_for is not None else os.path.join(
                PROJECT_ROOT, 'output', 'oastid', f'{args.source.lower()}_{args.tag}')
            os.makedirs(snap_dir, exist_ok=True)
            snap_path = os.path.join(snap_dir, f'epoch_{epoch:03d}.pt')
            torch.save({
                'state_dict': {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                },
                'epoch': int(epoch),
                'stage1_val': float(val_mae),
                'args': vars(args),
                'kind': 'stage1_epoch',
            }, snap_path)
            print(f'[stage1] snapshot {snap_path} val_mae={val_mae:.4f}', flush=True)
        if val_mae < best_val - 1e-4:
            best_val, bad = val_mae, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if save_every <= 0 and bad >= args.patience:
                print(f'[stage1] early stop (best={best_val:.4f})', flush=True)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


# =============================== Stage 2 ERM ================================== #
def meta_train(args, model, device, Xtr, Ytr, Itr, A, val_loader, scaler):
    """Stage2: episodic training over RANDOM SOURCE SUBGRAPHS (the OAGNN
    'random-sample' augmentation).

    Each meta-iteration samples ``task_num`` random node-subgraphs of the source
    graph (sizes from ``build_sizes``); for each we recompute the Laplacian-
    eigenmap F and predict a random time batch, accumulating a normalized-space
    Huber, then take one Adam meta-step. This trains the SINGLE shared model to
    forecast on graphs of many sizes/topologies -> improves size-generalization
    for zero-shot transfer, while staying stable (unlike bilevel MAML, which is
    ill-posed here because Laplacian eigenvectors carry sign/basis ambiguity that
    makes cross-subgraph second-order meta-gradients noisy).

    The best source-val checkpoint is kept, seeded with the Stage1 init so Stage2
    can never regress below Stage1.
    """
    node_max = int(min(args.node_max, A.shape[0]))
    node_min = int(min(args.node_min, max(node_max - 1, 1)))
    val_ei, val_ew, val_nf = build_graph(A, node_pe_dim=args.node_pe_dim, device=device)
    sem_full = load_sem_emb(args, args.source, device)
    behav_full = load_behav_pi(args, args.source, device)
    behav_meta = build_behav_meta_context(args, A, sem_full, behav_full, Itr)
    opt = torch.optim.Adam(model.parameters(), lr=args.meta_lr, weight_decay=args.weight_decay)
    ute = args.use_time_emb
    cal_start, spd, mps = calendar_of(args.source, getattr(args, 'calendar_start', None))
    tsz = model.time_of_day_size
    permute_adj = _stage2_permute_adj_flag(args)
    phase_exchange = _stage2_phase_exchange_flag(args)
    period_len = _stage2_period_len(args, spd)
    phase_mode = _stage2_phase_exchange_mode(args)
    phase_week_align = _stage2_phase_week_align_flag(args)
    d2 = _huber_delta_stage2(args)

    def src_val_mae():
        model.eval()
        with torch.no_grad():
            vl, vb = 0.0, 0
            for b in val_loader:
                x = b[0][..., :args.input_dim].float().to(device)
                y = b[1][..., :args.output_dim].float().to(device)
                tid, dow = (time_indices(
                    b[2], spd, tsz, device,
                    calendar_start=cal_start, minutes_per_step=mps) if ute else (None, None))
                vl += masked_mae_real(
                    model(x, val_ei, val_ew, val_nf, tid, dow, sem_full, behav_full),
                    y, scaler, args.mask_value).item()
                vb += 1
        model.train()
        return vl / max(vb, 1)

    # seed with the stage1 init so stage2 can never end up worse
    # Warm-start FROM Stage1, but do NOT keep Stage1 as a checkpoint candidate
    # (no rollback to Stage1). Best = Stage2's own best src-val during Stage2.
    s1_init = src_val_mae()
    best_val, best_state = float('inf'), None
    print(f'[stage2/ERM] warm-start src_val_mae={s1_init:.4f} (no Stage1 rollback) '
          f'permute_adj={int(permute_adj)} phase_exchange={int(phase_exchange)} '
          f'phase_preprocess={int(_stage2_phase_preprocess_flag(args))} '
          f'period_len={period_len} phase_mode={phase_mode} '
          f'week_align={int(phase_week_align)} '
          f'huber_delta_stage2={d2}', flush=True)

    N = A.shape[0]
    stage2_lambda = float(getattr(args, 'stage2_lambda', 1.0))
    print(f'[stage2/ERM] stage2_lambda={stage2_lambda:g} '
          f'(L=(L_full + λ Σ L_sub)/(1+λ n_sub))', flush=True)
    cross_packs = {}
    for it in range(1, args.meta_iters + 1):
        # always anchor on the FULL source graph (task 0) + random subgraphs, so
        # full-graph performance is directly optimized and cannot drift away while
        # the subgraphs add size/topology augmentation.
        sizes = [N] + build_sizes(num=max(args.task_num - 1, 1), min_n=node_min, max_n=node_max)
        model.train()
        opt.zero_grad()
        full_loss, sub_sum, n_sub = None, None, 0
        for s in sizes:
            # Full-graph anchor (s==N): keep original order (no permute) so Stage2
            # still directly optimizes the true source labeling. Random subgraphs
            # (s < N) optionally get STUNet-style adjacency permutation.
            # PhaseFormer phase-exchange is temporal (not spatial): apply on every
            # Stage2 task including the full-graph anchor when enabled.
            do_perm = bool(permute_adj and int(s) < int(N))
            xs, ys, xq, yq, ei, ew, nf, tid, dow, sem, behav = sample_task(
                Xtr, Ytr, Itr, A, s, args.k_spt, args.k_qry, args.node_pe_dim, device,
                spd=spd, table_size=tsz, calendar_start=cal_start, minutes_per_step=mps,
                permute_adj=do_perm,
                phase_exchange=phase_exchange, period_len=period_len,
                phase_exchange_mode=phase_mode, phase_week_align=phase_week_align,
                sem_full=sem_full, behav_full=behav_full, behav_meta=behav_meta)
            xin = torch.cat([xs, xq], 0)
            yin = torch.cat([ys, yq], 0)  # normalized targets
            pred = model(xin, ei, ew, nf, tid, dow, sem, behav)
            loss_i = huber_norm(pred, yin, d2)
            if int(s) == int(N) and full_loss is None:
                full_loss = loss_i
            else:
                sub_sum = loss_i if sub_sum is None else (sub_sum + loss_i)
                n_sub += 1
        if full_loss is None:
            raise RuntimeError('Stage2 ERM expected a full-graph anchor task')
        if n_sub == 0 or stage2_lambda == 0.0:
            total = full_loss
        else:
            total = (full_loss + stage2_lambda * sub_sum) / (1.0 + stage2_lambda * n_sub)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()

        if it % args.meta_eval_every == 0 or it == args.meta_iters:
            val_mae = src_val_mae()
            print(f'[stage2/ERM] iter {it:04d}/{args.meta_iters} '
                  f'subgraph_huber(norm)={float(total.detach()):.4f} '
                  f'lambda={stage2_lambda:g} src_val_mae={val_mae:.4f}', flush=True)
            if val_mae < best_val - 1e-4:
                best_val = val_mae
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        cross_packs = maybe_run_stage2_cross(args, model, device, it, cross_packs)
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_val = src_val_mae()
    return model, best_val


# ===================== Stage 2 (OAGNN faithful MAML) ===================== #
def meta_train_maml(args, model, device, Xtr, Ytr, Itr, A, val_loader, scaler):
    """Stage2 = OAGNN's first-order MAML (faithful port of subgraphlearning.Meta).

    Per meta-iter: sample ``task_num`` random source subgraphs; for each, run an
    inner loop of ``update_step`` SGD steps on the SUPPORT set (normalized Huber,
    functional_call fast-weights), then the QUERY Huber with the adapted weights is
    the task meta-loss. Average task meta-losses -> one outer Adam step (grad-norm
    clipped). ``second_order`` toggles FOMAML vs full MAML. Inner/meta lr follow
    OAGNN's computeLR schedule. Best source-val checkpoint kept (seed = Stage1).
    """
    N = A.shape[0]
    node_max = int(min(args.node_max, N))
    node_min = int(min(args.node_min, max(node_max - 1, 1)))
    val_ei, val_ew, val_nf = build_graph(A, node_pe_dim=args.node_pe_dim, device=device)
    sem_full = load_sem_emb(args, args.source, device)
    behav_full = load_behav_pi(args, args.source, device)
    behav_meta = build_behav_meta_context(args, A, sem_full, behav_full, Itr)
    meta_opt = torch.optim.Adam(model.parameters(), lr=args.meta_lr, eps=1e-8, weight_decay=0)
    ute = args.use_time_emb
    cal_start, spd, mps = calendar_of(args.source, getattr(args, 'calendar_start', None))
    tsz = model.time_of_day_size
    inner_max, inner_min = args.update_lr, args.update_lr * 0.7
    meta_max, meta_min = args.meta_lr, args.meta_lr * 0.7
    d2 = _huber_delta_stage2(args)

    def src_val_mae():
        model.eval()
        with torch.no_grad():
            vl, vb = 0.0, 0
            for b in val_loader:
                x = b[0][..., :args.input_dim].float().to(device)
                y = b[1][..., :args.output_dim].float().to(device)
                tid, dow = (time_indices(
                    b[2], spd, tsz, device,
                    calendar_start=cal_start, minutes_per_step=mps) if ute else (None, None))
                vl += masked_mae_real(
                    model(x, val_ei, val_ew, val_nf, tid, dow, sem_full, behav_full),
                    y, scaler, args.mask_value).item()
                vb += 1
        model.train()
        return vl / max(vb, 1)

    # Warm-start FROM Stage1, but do NOT keep Stage1 as a checkpoint candidate
    # (no rollback to Stage1). Best = Stage2's own best src-val during Stage2.
    s1_init = src_val_mae()
    best_val, best_state = float('inf'), None
    permute_adj = _stage2_permute_adj_flag(args)
    phase_exchange = _stage2_phase_exchange_flag(args)
    period_len = _stage2_period_len(args, spd)
    phase_mode = _stage2_phase_exchange_mode(args)
    phase_week_align = _stage2_phase_week_align_flag(args)
    print(f'[stage2/MAML] warm-start src_val_mae={s1_init:.4f} (no Stage1 rollback) '
          f'second_order={bool(args.second_order)} update_step={args.update_step} '
          f'permute_adj={int(permute_adj)} phase_exchange={int(phase_exchange)} '
          f'phase_preprocess={int(_stage2_phase_preprocess_flag(args))} '
          f'period_len={period_len} phase_mode={phase_mode} '
          f'week_align={int(phase_week_align)} '
          f'huber_delta_stage2={d2}', flush=True)

    def fwd(params, x, ei, ew, nf, tid, dow, sem, behav=None):
        return functional_call(model, params, args=(x, ei, ew, nf, tid, dow, sem, behav))

    patience = 0
    cross_packs = {}
    for it in range(1, args.meta_iters + 1):
        for g in meta_opt.param_groups:
            g['lr'] = _compute_lr(it, args.meta_iters, meta_min, meta_max)
        inner_lr = _compute_lr(it, args.meta_iters, inner_min, inner_max)
        sizes = build_sizes(num=max(args.task_num, 1), min_n=node_min, max_n=node_max)

        model.train()
        meta_opt.zero_grad()
        outer = torch.zeros((), device=device)
        qsum = 0.0
        for s in sizes:
            # MAML tasks are all random subgraphs (incl. possible s==N). Apply
            # STUNet-style adj permutation whenever enabled so node order is not
            # a spurious cue across meta-tasks. PhaseFormer phase-exchange remixes
            # traffic inside phase buckets (support/query separately).
            xs, ys, xq, yq, ei, ew, nf, tid, dow, sem, behav = sample_task(
                Xtr, Ytr, Itr, A, s, args.k_spt, args.k_qry, args.node_pe_dim, device,
                spd=spd, table_size=tsz, calendar_start=cal_start, minutes_per_step=mps,
                permute_adj=permute_adj,
                phase_exchange=phase_exchange, period_len=period_len,
                phase_exchange_mode=phase_mode, phase_week_align=phase_week_align,
                sem_full=sem_full, behav_full=behav_full, behav_meta=behav_meta)
            ns = xs.shape[0]
            tid_s = tid[:ns] if tid is not None else None
            dow_s = dow[:ns] if dow is not None else None
            tid_q = tid[ns:] if tid is not None else None
            dow_q = dow[ns:] if dow is not None else None

            fast = OrderedDict(model.named_parameters())
            for _k in range(args.update_step):
                ps = fwd(fast, xs, ei, ew, nf, tid_s, dow_s, sem, behav)
                loss_s = huber_norm(ps, ys, d2)          # normalized Huber (was MAE)
                grads = torch.autograd.grad(
                    loss_s, fast.values(),
                    create_graph=bool(args.second_order),
                    retain_graph=bool(args.second_order),
                    allow_unused=False)
                fast = OrderedDict(
                    (n, p - inner_lr * g) for (n, p), g in zip(fast.items(), grads))
            pq = fwd(fast, xq, ei, ew, nf, tid_q, dow_q, sem, behav)
            loss_q = huber_norm(pq, yq, d2)
            outer = outer + loss_q
            qsum += float(loss_q)
        outer = outer / len(sizes)
        outer.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        meta_opt.step()

        cross_packs = maybe_run_stage2_cross(args, model, device, it, cross_packs)
        if it % args.meta_eval_every == 0 or it == args.meta_iters:
            vm = src_val_mae()
            print(f'[stage2/MAML] iter {it:04d}/{args.meta_iters} '
                  f'query_huber(norm)={qsum/len(sizes):.4f} src_val_mae={vm:.4f} '
                  f'inner_lr={inner_lr:.2e}', flush=True)
            if vm < best_val - 1e-4:
                best_val, patience = vm, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= args.meta_patience:
                    print(f'[stage2/MAML] early stop (best={best_val:.4f})', flush=True)
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_val = src_val_mae()
    return model, best_val


# ===================== Stage 2 (Structure-MLDG / Orbit-MLDG) ===================== #
def meta_train_mldg(args, model, device, Xtr, Ytr, Itr, A, val_loader, scaler):
    """Stage2 = Structure-MLDG (Orbit-MLDG) on source subgraphs.

    Extends AAAI'18 MLDG (Li et al.): virtual meta-train / meta-test domains are
    *disjoint induced subgraphs* of the source city (structural domain shift with
    recomputed Z=g(A)), not same-graph support/query splits as in MAML.

    Per meta-iter, for each of ``task_num`` domain pairs (S_v, T_v):
      L = L_Sv(θ) + β L_Tv(θ') + γ L_Tv(θ) [+ δ L_full(θ)]
    where θ' = θ - α ∇ L_Sv(θ) (FOMAML-style one/few steps). The γ term aligns
    training with strict zero-shot Cross (deploy θ with no adaptation). Noroll:
    best = Stage2's own best clean source-val (no Stage1 rollback).

    Every L_* term is normalized-space Huber (``huber_delta_stage2``). The MLDG
    coefficient ``δ`` is still ``args.mldg_delta`` (full-graph weight), not the
    Huber threshold.
    """
    N = A.shape[0]
    node_max = int(min(args.node_max, N))
    node_min = int(min(args.node_min, max(node_max - 1, 1)))
    val_ei, val_ew, val_nf = build_graph(A, node_pe_dim=args.node_pe_dim, device=device)
    sem_full = load_sem_emb(args, args.source, device)
    meta_opt = torch.optim.Adam(model.parameters(), lr=args.meta_lr, eps=1e-8, weight_decay=0)
    ute = args.use_time_emb
    cal_start, spd, mps = calendar_of(args.source, getattr(args, 'calendar_start', None))
    tsz = model.time_of_day_size
    inner_max, inner_min = args.update_lr, args.update_lr * 0.7
    meta_max, meta_min = args.meta_lr, args.meta_lr * 0.7
    beta = float(getattr(args, 'mldg_beta', 1.0))
    gamma = float(getattr(args, 'mldg_gamma', 1.0))
    delta = float(getattr(args, 'mldg_delta', 0.5))
    k_batch = int(getattr(args, 'k_spt', 128)) + int(getattr(args, 'k_qry', 128))
    # MLDG paper uses one meta-train step; keep update_step but default scripts use 1.
    update_step = int(max(getattr(args, 'update_step', 1), 1))
    permute_adj = _stage2_permute_adj_flag(args)
    prefer_larger_tgt = bool(int(getattr(args, 'mldg_prefer_larger_target', 1)))
    d2 = _huber_delta_stage2(args)

    def src_val_mae():
        model.eval()
        with torch.no_grad():
            vl, vb = 0.0, 0
            for b in val_loader:
                x = b[0][..., :args.input_dim].float().to(device)
                y = b[1][..., :args.output_dim].float().to(device)
                tid, dow = (time_indices(
                    b[2], spd, tsz, device,
                    calendar_start=cal_start, minutes_per_step=mps) if ute else (None, None))
                vl += masked_mae_real(model(x, val_ei, val_ew, val_nf, tid, dow, sem_full), y, scaler, args.mask_value).item()
                vb += 1
        model.train()
        return vl / max(vb, 1)

    s1_init = src_val_mae()
    best_val, best_state = float('inf'), None
    print(f'[stage2/MLDG] warm-start src_val_mae={s1_init:.4f} (no Stage1 rollback) '
          f'second_order={bool(args.second_order)} update_step={update_step} '
          f'beta={beta} gamma={gamma} delta={delta} k_batch={k_batch} '
          f'prefer_larger_target={int(prefer_larger_tgt)} '
          f'huber_delta_stage2={d2}', flush=True)

    def fwd(params, x, ei, ew, nf, tid, dow, sem):
        return functional_call(model, params, args=(x, ei, ew, nf, tid, dow, sem))

    # Full-graph anchor batch indices (resampled lightly each time we need it).
    def sample_full_batch():
        return sample_subgraph_batch(
            Xtr, Ytr, Itr, A, np.arange(N), k_batch, args.node_pe_dim, device,
            spd=spd, table_size=tsz, calendar_start=cal_start, minutes_per_step=mps,
            permute_adj=False, sem_full=sem_full)

    patience = 0
    cross_packs = {}
    for it in range(1, args.meta_iters + 1):
        for g in meta_opt.param_groups:
            g['lr'] = _compute_lr(it, args.meta_iters, meta_min, meta_max)
        inner_lr = _compute_lr(it, args.meta_iters, inner_min, inner_max)
        n_pairs = max(int(args.task_num), 1)

        model.train()
        meta_opt.zero_grad()
        outer = torch.zeros((), device=device)
        ls_sum = lt_ad_sum = lt_z_sum = lf_sum = 0.0

        for _ in range(n_pairs):
            sv, tv, _ns, _nt = sample_mldg_domain_pair(
                Xtr, Ytr, Itr, A, node_min, node_max, k_batch, args.node_pe_dim, device,
                spd=spd, table_size=tsz, calendar_start=cal_start, minutes_per_step=mps,
                permute_adj=permute_adj, prefer_larger_target=prefer_larger_tgt,
                sem_full=sem_full)
            xs, ys, ei_s, ew_s, nf_s, tid_s, dow_s, sem_s = sv
            xt, yt, ei_t, ew_t, nf_t, tid_t, dow_t, sem_t = tv

            fast = OrderedDict(model.named_parameters())
            # Meta-train on S_v → θ'
            loss_s = None
            for _k in range(update_step):
                ps = fwd(fast, xs, ei_s, ew_s, nf_s, tid_s, dow_s, sem_s)
                loss_s = huber_norm(ps, ys, d2)
                grads = torch.autograd.grad(
                    loss_s, fast.values(),
                    create_graph=bool(args.second_order),
                    retain_graph=bool(args.second_order),
                    allow_unused=False)
                fast = OrderedDict(
                    (n, p - inner_lr * g) for (n, p), g in zip(fast.items(), grads))
            # θ terms on S_v / T_v / full (zero-shot aligned)
            # Recompute L_Sv(θ) with current model params (not fast weights).
            theta = OrderedDict(model.named_parameters())
            loss_s_theta = huber_norm(
                fwd(theta, xs, ei_s, ew_s, nf_s, tid_s, dow_s, sem_s), ys, d2)
            loss_t_adapt = huber_norm(
                fwd(fast, xt, ei_t, ew_t, nf_t, tid_t, dow_t, sem_t), yt, d2)
            loss_t_zero = huber_norm(
                fwd(theta, xt, ei_t, ew_t, nf_t, tid_t, dow_t, sem_t), yt, d2)

            pair_loss = loss_s_theta + beta * loss_t_adapt + gamma * loss_t_zero
            if delta > 0:
                xf, yf, ei_f, ew_f, nf_f, tid_f, dow_f, sem_f = sample_full_batch()
                loss_full = huber_norm(
                    fwd(theta, xf, ei_f, ew_f, nf_f, tid_f, dow_f, sem_f), yf, d2)
                pair_loss = pair_loss + delta * loss_full
                lf_sum += float(loss_full)
            else:
                loss_full = None

            outer = outer + pair_loss
            ls_sum += float(loss_s_theta)
            lt_ad_sum += float(loss_t_adapt)
            lt_z_sum += float(loss_t_zero)

        outer = outer / n_pairs
        outer.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        meta_opt.step()

        cross_packs = maybe_run_stage2_cross(args, model, device, it, cross_packs)
        if it % args.meta_eval_every == 0 or it == args.meta_iters:
            vm = src_val_mae()
            print(f'[stage2/MLDG] iter {it:04d}/{args.meta_iters} '
                  f'Ls={ls_sum/n_pairs:.4f} Lt\'={lt_ad_sum/n_pairs:.4f} '
                  f'Lt0={lt_z_sum/n_pairs:.4f} Lf={lf_sum/max(n_pairs,1):.4f} '
                  f'src_val_mae={vm:.4f} inner_lr={inner_lr:.2e}', flush=True)
            if vm < best_val - 1e-4:
                best_val, patience = vm, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= args.meta_patience:
                    print(f'[stage2/MLDG] early stop (best={best_val:.4f})', flush=True)
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_val = src_val_mae()
    return model, best_val


def parse_args():
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument('--huber_delta_stage1', type=float, default=23.0,
                       help='Huber δ for Stage1 (original / inverse-std scale)')
    extra.add_argument('--huber_delta_stage2', type=float, default=0.15,
                       help='Huber δ for Stage2 ERM/MAML/MLDG (normalized scale)')
    extra.add_argument('--save_every', type=int, default=0,
                       help='Stage1: save current weights every N epochs (0=off). '
                            'Disables Stage1 early-stop so snapshots can reach stage1_epochs.')
    extra.add_argument('--save_from', type=int, default=40,
                       help='Stage1: first epoch eligible for --save_every snapshots')
    extra.add_argument('--stage2_lambda', type=float, default=1.0,
                       help='Stage2 ERM: L = (L_full + λ Σ L_sub) / (1 + λ n_sub). '
                            'λ=1 matches the previous equal-task average.')
    known, rest = extra.parse_known_args()
    argv_bak = list(sys.argv)
    sys.argv = [argv_bak[0]] + rest
    try:
        args = base.parse_args()
    finally:
        sys.argv = argv_bak
    args.huber_delta_stage1 = float(known.huber_delta_stage1)
    args.huber_delta_stage2 = float(known.huber_delta_stage2)
    args.save_every = int(known.save_every)
    args.save_from = int(known.save_from)
    args.stage2_lambda = float(known.stage2_lambda)
    return args


def _args_from_ckpt(cli_args, ckpt_args):
    merged = base._args_from_ckpt(cli_args, ckpt_args)
    if not hasattr(merged, 'huber_delta_stage1'):
        merged.huber_delta_stage1 = 23.0
    if not hasattr(merged, 'huber_delta_stage2'):
        merged.huber_delta_stage2 = 0.15
    if not hasattr(merged, 'stage2_lambda'):
        merged.stage2_lambda = 1.0
    return merged


def main():
    cli_args = parse_args()
    torch.manual_seed(cli_args.seed)
    np.random.seed(cli_args.seed)

    # -------- Cross-only from checkpoint --------
    if cli_args.eval_ckpt:
        ckpt_path = cli_args.eval_ckpt
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(PROJECT_ROOT, ckpt_path)
        print(f'[eval_ckpt] loading {ckpt_path}', flush=True)
        ckpt = torch.load(ckpt_path, map_location='cpu')
        args = _resolve_sem_flag(_args_from_ckpt(cli_args, ckpt['args']))
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        targets = [t.strip() for t in args.targets.split(',') if t.strip()]
        base._SE_TYPE = args.se_type
        base._SE_SELF_LOOP = None if int(getattr(args, 'se_self_loop', -1)) < 0 else bool(args.se_self_loop)
        print(f'OA-STID HUBER CROSS-ONLY | source={args.source} -> {targets} | h={args.horizon} '
              f'device={device} fusion={args.fusion} attn_node_chunk={args.attn_node_chunk} '
              f'tag={args.tag}', flush=True)
        model = build_oastid_model(args).to(device)
        missing, unexpected = model.load_state_dict(ckpt['state_dict'], strict=False)
        print(f'[eval_ckpt] loaded stage1_val={ckpt.get("stage1_val")} '
              f'stage2_val={ckpt.get("stage2_val")} '
              f'missing={len(missing)} unexpected={len(unexpected)}', flush=True)
        if missing:
            print(f'[eval_ckpt] missing keys (first 10): {missing[:10]}', flush=True)
        if unexpected:
            print(f'[eval_ckpt] unexpected keys (first 10): {unexpected[:10]}', flush=True)
        print(f'model params: {sum(p.numel() for p in model.parameters())}', flush=True)
        print('\n########## STAGE 3: cross-domain zero-shot (from ckpt) ##########', flush=True)
        summary = {tgt: eval_target(args, model, device, tgt) for tgt in targets}
        print('\n================ SUMMARY (cross-only) ================', flush=True)
        for tgt, m in summary.items():
            print(f'{args.source} -> {tgt}: MAE={m["MAE"]:.4f} RMSE={m["RMSE"]:.4f} '
                  f'Masked-MAPE={m["Masked-MAPE%"]:.4f}%', flush=True)
        return

    args = _resolve_sem_flag(cli_args)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    targets = [t.strip() for t in args.targets.split(',') if t.strip()]
    base._SE_TYPE = args.se_type
    base._SE_SELF_LOOP = None if args.se_self_loop < 0 else bool(args.se_self_loop)
    print(f'OA-STID HUBER | source={args.source} -> {targets} | h={args.horizon} '
          f'device={device} | se_type={base._SE_TYPE} se_self_loop={base._SE_SELF_LOOP} '
          f'fusion={args.fusion} '
          f'huber_delta_stage1={_huber_delta_stage1(args)} '
          f'huber_delta_stage2={_huber_delta_stage2(args)}', flush=True)

    model = build_oastid_model(args).to(device)
    print(f'use_time_emb={args.use_time_emb} use_sem_emb={args.use_sem_emb} '
          f'sem_dim={getattr(args, "sem_dim", 32)} '
          f'use_bsts={getattr(args, "use_bsts", 1)} '
          f'stats_v2={getattr(args, "stats_v2", 1)} '
          f'stats_nbr={getattr(args, "stats_nbr", 1)} '
          f'stats_rank={getattr(args, "stats_rank", 1)} '
          f'stats_spec={getattr(args, "stats_spec", 1)} '
          f'stats_proj={getattr(args, "stats_proj_temp_dim", 16)}/'
          f'{getattr(args, "stats_proj_spat_dim", 8)}/'
          f'{getattr(args, "stats_proj_graph_dim", 8)}/'
          f'{getattr(args, "stats_proj_spec_dim", 8)} fusion={args.fusion} '
          f'residual_base={getattr(args, "use_residual_base", 1)}', flush=True)
    if args.use_time_emb:
        src_cal, _, src_mps = calendar_of(args.source, args.calendar_start)
        print(f'[time] weekday=REAL calendar (Mon=0..Sun=6); '
              f'source start={src_cal} min/step={src_mps}', flush=True)
        for t in targets:
            tc, _, tm = calendar_of(t)
            print(f'[time] target {t} start={tc} min/step={tm}', flush=True)
    print(f'model params: {sum(p.numel() for p in model.parameters())}', flush=True)

    print('\n########## STAGE 1: source pretrain (Huber) ##########', flush=True)
    t0 = time.perf_counter()
    s1_ckpt = str(getattr(args, 'stage1_ckpt', '') or '').strip()
    if s1_ckpt or int(getattr(args, 'skip_stage1', 0)):
        if not s1_ckpt:
            raise ValueError('--skip_stage1=1 requires --stage1_ckpt')
        loader = load_compatible_state_dict
        if loader is None:
            loader = base.load_compatible_state_dict
        ckpt_path = s1_ckpt
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(PROJECT_ROOT, ckpt_path)
        extra = loader(model, ckpt_path)
        if extra is None:
            raw = torch.load(ckpt_path, map_location='cpu')
            extra = raw.get('stage1_val') if isinstance(raw, dict) else None
        s1_val = float(extra) if extra is not None else float('nan')
        print(
            f'[stage1] SKIPPED train; loaded {ckpt_path} stage1_val={s1_val} '
            f'({time.perf_counter()-t0:.1f}s)',
            flush=True,
        )
    else:
        model, s1_val = train_source(args, model, device)
        print(f'[stage1] done best_val_mae={s1_val:.4f} ({time.perf_counter()-t0:.1f}s)', flush=True)

    if args.skip_stage2:
        print('\n########## STAGE 2: SKIPPED (stage1-only weights) ##########', flush=True)
        s2_val = s1_val
    else:
        print(f'\n########## STAGE 2: {args.stage2_mode.upper()} (Huber) ##########', flush=True)
        t0 = time.perf_counter()
        Xtr, Ytr, Itr, val_loader, scaler, A = collect_source_arrays(args, device)
        print(f'[stage2] windows X={tuple(Xtr.shape)} nodes={A.shape[0]} '
              f'subgraph=[{args.node_min},{min(args.node_max,A.shape[0])}] '
              f'task_num={args.task_num} update_step={args.update_step} '
              f'permute_adj={int(getattr(args, "stage2_permute_adj", 0))} '
              f'phase_exchange={int(getattr(args, "stage2_phase_exchange", 0))} '
              f'phase_preprocess={int(getattr(args, "stage2_phase_preprocess", 0))} '
              f'period_len={int(getattr(args, "stage2_period_len", 0))} '
              f'phase_mode={getattr(args, "stage2_phase_exchange_mode", "both")} '
              f'week_align={int(getattr(args, "stage2_phase_week_align", 1))} '
              f'mldg_beta={getattr(args, "mldg_beta", 1.0)} '
              f'mldg_gamma={getattr(args, "mldg_gamma", 1.0)} '
              f'mldg_delta={getattr(args, "mldg_delta", 0.5)} '
              f'huber_delta_stage2={_huber_delta_stage2(args)}',
              flush=True)
        if args.stage2_mode == 'maml':
            run_stage2 = meta_train_maml
        elif args.stage2_mode == 'mldg':
            run_stage2 = meta_train_mldg
        else:
            run_stage2 = meta_train
        model, s2_val = run_stage2(args, model, device, Xtr, Ytr, Itr, A, val_loader, scaler)
        print(f'[stage2] done best_src_val_mae={s2_val:.4f} ({time.perf_counter()-t0:.1f}s)', flush=True)

    ckpt_dir = os.path.join(PROJECT_ROOT, 'output', 'oastid', f'{args.source.lower()}_{args.tag}')
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'stage1_val': s1_val,
                'stage2_val': s2_val, 'args': vars(args)}, os.path.join(ckpt_dir, 'best.pt'))

    print('\n########## STAGE 3: cross-domain zero-shot ##########', flush=True)
    summary = {tgt: eval_target(args, model, device, tgt) for tgt in targets}

    print('\n================ SUMMARY (stage1+stage2+cross, Huber train / MAE eval) ================', flush=True)
    for tgt, m in summary.items():
        print(f'{args.source} -> {tgt}: MAE={m["MAE"]:.4f} RMSE={m["RMSE"]:.4f} '
              f'Masked-MAPE={m["Masked-MAPE%"]:.4f}%', flush=True)


if __name__ == '__main__':
    main()
