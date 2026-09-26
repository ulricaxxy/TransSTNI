"""GraphSAGE orbit + Stage2 ERM/MAML helpers (full-neighbor SAGEConv).

Keeps the original ``code/src/model/graph_sage.py`` GraphSAGE intact
(3× SAGEConv + ``sample_neighbors``). Changes:

1. GraphSAGE neighbor aggregation is CLI-gated by ``--sage_full_neighbor``:
   * ``1`` (default): full-neighbor formula path
     (``dense_to_sparse(A)``) during train and eval.
   * ``0``: original limited-neighbor path — train-time
     ``sample_neighbors(A, k=sage_k)``, eval-time full edges.
2. Stage2 size sampling is CLI-gated:
   * ``--sage_stage2_fullgraph 0`` (default): random **node-subgraph ERM/MAML**
     with sizes in ``[--sage_subgraph_node_min, --sage_subgraph_node_max]``
     (default 300–700), clamped to source ``N``.
   * ``--sage_stage2_fullgraph 1``: previous full-graph path
     (``node_min=node_max=N``, ``task_num=1``).
3. Optional CLI ``--stage1_to_cross 1``: Stage1 → Cross (skip Stage2). Stage1
   keeps its full Huber train / MAE val / early-stop loop; every
   ``--stage1_cross_every`` epochs runs the same full zero-shot Cross as
   Stage2 mid-training Cross.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse

# Module flag flipped by parse_args after CLI is known.
# Default True preserves historical full-neighbor behaviour.
_SAGE_USE_FULL_NEIGHBOR = True


def _install_full_neighbor_forward():
    from lib import orbit_graphsage as og

    def forward(self, A, node_features):
        # Runtime switch: full-neighbor formula vs original limited sampling.
        use_full = bool(globals().get('_SAGE_USE_FULL_NEIGHBOR', True))
        if use_full:
            # Full-neighbor mean aggregation (formula). sample_neighbors remains.
            edge_index, _ = dense_to_sparse(A)
            self._full_edge_index = edge_index
        else:
            # Original GraphSAGE: train = sample_neighbors(k=sage_k); eval = full.
            if self.training:
                edge_index = self.sample_neighbors(A, k=self.args.sage_k)
            else:
                if self._full_edge_index is None:
                    self._full_edge_index, _ = dense_to_sparse(A)
                edge_index = self._full_edge_index
        x = self.conv1(node_features, edge_index)
        x = self.norm1(x) if self.sage_norm else x
        x = F.relu(x)
        x = F.dropout(x, p=self.sage_dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.norm2(x) if self.sage_norm else x
        x = F.relu(x)
        x = F.dropout(x, p=self.sage_dropout, training=self.training)

        x = self.conv3(x, edge_index)
        return x

    og.GraphSAGE.forward = forward
    print(
        '[sage/nosubgraph] GraphSAGE.forward installed '
        '(default full-neighbor; --sage_full_neighbor 0 → train-time '
        'sample_neighbors(k=sage_k); Stage2 subgraph sampling gated separately)',
        flush=True,
    )


def _force_full_graph_args(args, A, tag='[sage/stage2]'):
    n = int(A.shape[0])
    old = (int(getattr(args, 'node_min', -1)),
           int(getattr(args, 'node_max', -1)),
           int(getattr(args, 'task_num', -1)))
    args.node_min = n
    args.node_max = n
    args.task_num = 1
    print(
        f'{tag} Stage2 full-graph: N={n} '
        f'node_min/max/task_num {old} → ({n},{n},1)',
        flush=True,
    )
    return args


def _force_random_subgraph_args(args, A, tag='[sage/stage2]'):
    """Random source node-subgraph ERM/MAML: sizes in [lo, hi] ∩ [1, N]."""
    n = int(A.shape[0])
    lo = int(getattr(args, 'sage_subgraph_node_min', 300))
    hi = int(getattr(args, 'sage_subgraph_node_max', 700))
    if lo > hi:
        lo, hi = hi, lo
    lo = max(1, min(lo, n))
    hi = max(1, min(hi, n))
    if lo > hi:
        lo = hi
    old = (int(getattr(args, 'node_min', -1)),
           int(getattr(args, 'node_max', -1)),
           int(getattr(args, 'task_num', -1)))
    # ``build_sizes`` uses ``np.random.randint(low, high)`` with **exclusive**
    # ``high``. Bump by +1 so CLI [lo, hi] is an inclusive node-count range.
    args.node_min = lo
    args.node_max = min(n, hi) + 1
    # Keep caller's task_num (multi-subgraph ERM). Only repair invalid values.
    task_num = int(getattr(args, 'task_num', 1))
    if task_num < 1:
        task_num = 1
        args.task_num = task_num
    print(
        f'{tag} Stage2 random-subgraph ERM/MAML: N={n} '
        f'node_min/max/task_num {old} → '
        f'({args.node_min},{args.node_max},{int(args.task_num)}) '
        f'(inclusive size [{lo},{hi}]; build_sizes high-exclusive)',
        flush=True,
    )
    return args


def _apply_stage2_graph_args(args, A):
    if int(getattr(args, 'sage_stage2_fullgraph', 0)) == 1:
        return _force_full_graph_args(args, A)
    return _force_random_subgraph_args(args, A)


def _install_stage2_fullgraph_wrappers(huber_module):
    """Wrap ERM / MAML Stage2: full-graph OR random subgraph (CLI)."""
    _orig_erm = huber_module.meta_train
    _orig_maml = huber_module.meta_train_maml

    def meta_train(args, model, device, Xtr, Ytr, Itr, A, *rest, **kwargs):
        _apply_stage2_graph_args(args, A)
        return _orig_erm(args, model, device, Xtr, Ytr, Itr, A, *rest, **kwargs)

    def meta_train_maml(args, model, device, Xtr, Ytr, Itr, A, *rest, **kwargs):
        _apply_stage2_graph_args(args, A)
        return _orig_maml(args, model, device, Xtr, Ytr, Itr, A, *rest, **kwargs)

    huber_module.meta_train = meta_train
    huber_module.meta_train_maml = meta_train_maml
    print(
        '[sage/stage2] meta_train/meta_train_maml patched: '
        'default random-subgraph (node_min/max via CLI); '
        '--sage_stage2_fullgraph 1 restores full-graph ERM',
        flush=True,
    )


def maybe_run_stage1_cross(args, model, device, epoch, packs, huber_module):
    """Every ``stage1_cross_every`` Stage1 epochs: full zero-shot eval on all targets.

    Mirrors ``maybe_run_stage2_cross`` (same ``load_target_eval_pack`` /
    ``eval_target`` / jsonl logging), but keys on Stage1 epoch and writes
    ``stage1_cross.jsonl``. Uses the current weights (not the best-so-far
    checkpoint). Target loaders / graphs / Qwen-sem are cached in ``packs``
    so only the first call reloads.
    """
    every = int(getattr(args, 'stage1_cross_every', 20))
    if every <= 0 or int(epoch) % every != 0:
        return packs
    targets = huber_module.parse_target_list(args)
    if not targets:
        return packs
    print(
        f'\n########## STAGE1 CROSS @ epoch {int(epoch):04d} ##########',
        flush=True,
    )
    was_training = model.training
    model.eval()
    row = {
        'epoch': int(epoch),
        'targets': {},
    }
    # load_target_eval_pack lives on oastid_full (aliased into maybe_run globals);
    # Huber bootstrap may not re-export it on the module object.
    load_pack = getattr(huber_module, 'load_target_eval_pack', None)
    if load_pack is None:
        load_pack = huber_module.maybe_run_stage2_cross.__globals__[
            'load_target_eval_pack']
    eval_target = huber_module.eval_target
    for tgt in targets:
        if tgt not in packs:
            packs[tgt] = load_pack(args, tgt, device, model=model)
        ov = eval_target(
            args, model, device, tgt, step=int(epoch), pack=packs[tgt])
        row['targets'][tgt] = ov
        print(
            f"[stage1/cross] epoch={int(epoch):04d} {args.source}->{tgt} "
            f"MAE={ov['MAE']:.4f} RMSE={ov['RMSE']:.4f} "
            f"Masked-MAPE={ov['Masked-MAPE%']:.4f}%",
            flush=True,
        )
    log_dir = os.path.join(
        huber_module.PROJECT_ROOT,
        'output',
        'oastid',
        f'{args.source.lower()}_{args.tag}',
    )
    os.makedirs(log_dir, exist_ok=True)
    try:
        with open(os.path.join(log_dir, 'stage1_cross.jsonl'), 'a') as f:
            f.write(json.dumps(row) + '\n')
    except Exception:
        pass
    if was_training:
        model.train()
    return packs


def _build_train_source_stage1_to_cross(huber_module):
    """Full Stage1 Huber loop (not simplified) + optional mid-Stage1 Cross.

    Faithful copy of ``oastid_full_huber.train_source`` control flow:
    loaders / graph / sem / behav / Adam+MultiStepLR / Huber train / MAE val /
    early stop / restore best. Extra: when ``stage1_cross_every > 0``, call
    ``maybe_run_stage1_cross`` after each epoch that hits the cadence.
    """
    make_loader_args = huber_module.make_loader_args
    calendar_of = huber_module.calendar_of
    get_dataloader = huber_module.get_dataloader
    load_graph = huber_module.load_graph
    load_sem_emb = huber_module.load_sem_emb
    load_behav_pi = huber_module.load_behav_pi
    _huber_delta_stage1 = huber_module._huber_delta_stage1
    time_indices = huber_module.time_indices
    to_dev = huber_module.to_dev
    masked_huber_real = huber_module.masked_huber_real
    masked_mae_real = huber_module.masked_mae_real

    def train_source(args, model, device):
        src_args = make_loader_args(args, args.source)
        ute = args.use_time_emb
        cal_start, spd, mps = calendar_of(
            args.source, getattr(args, 'calendar_start', None))
        train_loader, val_loader, _, scaler = get_dataloader(
            src_args,
            normalizer='std',
            tod=ute,
            dow=ute,
            weather=False,
            single=False,
            return_index=ute,
        )
        edge_index, edge_weight, node_feat = load_graph(
            args, args.source, device)
        sem = load_sem_emb(args, args.source, device)
        behav = load_behav_pi(args, args.source, device)
        d1 = _huber_delta_stage1(args)

        def tds(batch):
            if not ute:
                return None, None
            return time_indices(
                batch[2],
                spd,
                model.time_of_day_size,
                device,
                calendar_start=cal_start,
                minutes_per_step=mps,
            )

        opt = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.MultiStepLR(
            opt,
            milestones=[
                int(args.stage1_epochs * 0.5),
                int(args.stage1_epochs * 0.75),
            ],
            gamma=0.5,
        )
        print(
            f'[stage1/huber] train Huber on original scale delta={d1}'
            f' (val stays MAE / masked_mae_real)',
            flush=True,
        )
        print(
            f'[sage/nosubgraph] Stage1→Cross mode: max_epochs={int(args.stage1_epochs)} '
            f'patience={int(args.patience)} '
            f'stage1_cross_every={int(getattr(args, "stage1_cross_every", 0))} '
            f'skip_stage2={int(getattr(args, "skip_stage2", 0))}',
            flush=True,
        )

        best_val, best_state, bad = float('inf'), None, 0
        cross_packs = {}
        for epoch in range(1, int(args.stage1_epochs) + 1):
            model.train()
            t0, tr_loss, nb = time.perf_counter(), 0.0, 0
            for batch in train_loader:
                x = to_dev(batch[0], device)[..., :args.input_dim]
                y = to_dev(batch[1], device)[..., :args.output_dim]
                tid, dow = tds(batch)
                opt.zero_grad()
                loss = masked_huber_real(
                    model(
                        x, edge_index, edge_weight, node_feat,
                        tid, dow, sem, behav),
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
                        model(
                            x, edge_index, edge_weight, node_feat,
                            tid, dow, sem, behav),
                        y, scaler, args.mask_value).item()
                    vb += 1
            val_mae = vl / max(vb, 1)
            print(
                f'[stage1][{args.source}] epoch {epoch:03d} '
                f'train_huber={tr_loss / max(nb, 1):.4f} '
                f'val_mae={val_mae:.4f} ({time.perf_counter() - t0:.1f}s)',
                flush=True,
            )

            # Mid-Stage1 full zero-shot Cross (current weights), cadence by epoch.
            cross_packs = maybe_run_stage1_cross(
                args, model, device, epoch, cross_packs, huber_module)

            if val_mae < best_val - 1e-4:
                best_val, bad = val_mae, 0
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
            else:
                bad += 1
                if bad >= args.patience:
                    print(
                        f'[stage1] early stop (best={best_val:.4f})',
                        flush=True,
                    )
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        return model, best_val

    return train_source


def _install_stage1_to_cross(huber_module):
    """CLI-gated Stage1→Cross path; default keeps original Stage1+Stage2 ERM."""
    _orig_parse = huber_module.parse_args
    _orig_train_source = huber_module.train_source
    _train_source_s1c = _build_train_source_stage1_to_cross(huber_module)

    # Keep explicit handles so old Stage2-ERM path stays callable / inspectable.
    huber_module._sage_orig_train_source = _orig_train_source
    huber_module._sage_stage1_to_cross_train_source = _train_source_s1c
    huber_module.maybe_run_stage1_cross = (
        lambda args, model, device, epoch, packs:
        maybe_run_stage1_cross(
            args, model, device, epoch, packs, huber_module)
    )

    def parse_args():
        extra = argparse.ArgumentParser(
            add_help=False,
            description='sage Stage2 / Stage1→Cross CLI (optional)',
        )
        extra.add_argument(
            '--stage1_to_cross',
            type=int,
            default=0,
            help=(
                '1: Stage1→Cross only (skip Stage2). Stage1 early-stop kept; '
                'run full zero-shot Cross every --stage1_cross_every epochs. '
                '0: Stage1→Stage2 ERM/MAML→Cross.'
            ),
        )
        extra.add_argument(
            '--stage1_cross_every',
            type=int,
            default=20,
            help=(
                'Stage1→Cross mode: run full zero-shot Cross on --targets every '
                'N Stage1 epochs; 0 disables mid-Stage1 cross (final Stage3 '
                'Cross after Stage1 still runs when skip_stage2=1).'
            ),
        )
        extra.add_argument(
            '--sage_stage2_fullgraph',
            type=int,
            default=0,
            help=(
                'Stage2 graph sampling: 0 = random node-subgraph ERM/MAML with '
                'sizes in [sage_subgraph_node_min, sage_subgraph_node_max] '
                '(default); 1 = previous full-graph path (node_min=node_max=N, '
                'task_num=1).'
            ),
        )
        extra.add_argument(
            '--sage_subgraph_node_min',
            type=int,
            default=300,
            help='Stage2 random-subgraph: minimum subgraph node count (clamped to N).',
        )
        extra.add_argument(
            '--sage_subgraph_node_max',
            type=int,
            default=700,
            help='Stage2 random-subgraph: maximum subgraph node count (clamped to N).',
        )
        extra.add_argument(
            '--sage_full_neighbor',
            type=int,
            default=1,
            help=(
                'GraphSAGE neighbor aggregation: 1 = full-neighbor formula '
                '(default, previous behaviour); 0 = limited neighbors — '
                'train-time sample_neighbors(A, k=sage_k), eval-time full edges '
                '(original GraphSAGE).'
            ),
        )
        extra.add_argument(
            '--fullgraph_pe_slice',
            type=int,
            default=0,
            help=(
                '1: Stage2 subgraph tasks slice full-graph Laplacian PE F_full[S] '
                'instead of recomputing eigenmaps on each induced subgraph.'
            ),
        )
        if any(a in ('-h', '--help') for a in sys.argv[1:]):
            print(
                '\n===== sage extra (Stage2 subgraph / Stage1→Cross) =====',
                flush=True,
            )
            extra.print_help()
            print(
                '===== end sage extra =====\n',
                flush=True,
            )
        known, rest = extra.parse_known_args()
        argv_bak = list(sys.argv)
        sys.argv = [argv_bak[0]] + rest
        try:
            args = _orig_parse()
        finally:
            sys.argv = argv_bak

        args.stage1_to_cross = int(known.stage1_to_cross)
        args.stage1_cross_every = int(known.stage1_cross_every)
        args.sage_stage2_fullgraph = int(known.sage_stage2_fullgraph)
        args.sage_subgraph_node_min = int(known.sage_subgraph_node_min)
        args.sage_subgraph_node_max = int(known.sage_subgraph_node_max)
        args.sage_full_neighbor = int(known.sage_full_neighbor)
        args.fullgraph_pe_slice = int(known.fullgraph_pe_slice)
        global _SAGE_USE_FULL_NEIGHBOR
        _SAGE_USE_FULL_NEIGHBOR = (int(args.sage_full_neighbor) == 1)
        print(
            f'[sage] neighbor mode='
            f'{"full-neighbor" if _SAGE_USE_FULL_NEIGHBOR else "limited-neighbor"} '
            f'(sage_full_neighbor={int(args.sage_full_neighbor)}, '
            f'sage_k={int(getattr(args, "sage_k", 3))})',
            flush=True,
        )
        if int(args.stage1_to_cross) == 1:
            # Stage1→Cross: never enter Stage2, regardless of prior skip_stage2.
            args.skip_stage2 = 1
            print(
                '[sage] --stage1_to_cross=1 → skip_stage2=1, '
                f'stage1_epochs={int(args.stage1_epochs)}, '
                f'patience={int(args.patience)}, '
                f'stage1_cross_every={int(args.stage1_cross_every)}',
                flush=True,
            )
        else:
            mode = (
                'full-graph'
                if int(args.sage_stage2_fullgraph) == 1
                else (
                    f"random-subgraph "
                    f"[{int(args.sage_subgraph_node_min)},"
                    f"{int(args.sage_subgraph_node_max)}]"
                )
            )
            print(
                f'[sage] Stage2 mode={mode} '
                f'(sage_stage2_fullgraph={int(args.sage_stage2_fullgraph)})',
                flush=True,
            )
        return args

    def train_source(args, model, device):
        if int(getattr(args, 'stage1_to_cross', 0)) == 1:
            return _train_source_s1c(args, model, device)
        return _orig_train_source(args, model, device)

    huber_module.parse_args = parse_args
    huber_module.train_source = train_source
    print(
        '[sage] Stage1→Cross + Stage2 subgraph/fullgraph CLI installed '
        '(--stage1_to_cross / --sage_stage2_fullgraph / '
        '--sage_subgraph_node_min/max / --sage_full_neighbor); '
        'default Stage2 = random subgraph [300,700]; '
        'default neighbor = full (--sage_full_neighbor 0 for limited k=sage_k)',
        flush=True,
    )


def install_stage2_fullgraph(huber_module):
    """Install GraphSAGE full-neighbor + Stage2 subgraph/fullgraph + Stage1→Cross."""
    _install_full_neighbor_forward()
    _install_stage2_fullgraph_wrappers(huber_module)
    _install_stage1_to_cross(huber_module)
    from lib.fullgraph_pe_patch import install_fullgraph_pe_patch

    install_fullgraph_pe_patch(huber_module)
