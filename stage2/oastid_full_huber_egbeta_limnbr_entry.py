"""Slim entry for Eg=β + limnbr k=3 nobehav ablation.

Only wires what this package actually uses:
  * Huber Stage1/2 (ERM) + freezesem
  * GraphSAGE limited-neighbor + Stage2 random subgraph
  * ``--no_beta_net`` / ``--skip_stage3`` / ``--spatial_avwgcn`` (default 0)
  * seed + require-GPU

Does **not** load behav A1/A3/B6 / cal-prior / AVWGCN-on / road-text-v3 /
spatial-z-domain-norm extras. Cross metrics come from ``--eval_ckpt`` only
(train passes ``--skip_stage3 1``).
"""
from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_EXTRA = argparse.ArgumentParser(add_help=False)
_EXTRA.add_argument('--no_beta_net', type=int, default=0, choices=(0, 1))
_EXTRA.add_argument('--skip_stage3', type=int, default=1, choices=(0, 1),
                    help='1=skip inline Stage3 after train (use eval_ckpt)')
_EXTRA.add_argument('--spatial_avwgcn', type=int, default=0, choices=(0, 1),
                    help='must stay 0 for Eg=β concat-Z')
_KNOWN, _REST = _EXTRA.parse_known_args()
sys.argv = [sys.argv[0]] + _REST

from lib.nobeta_orbit_patch import install_nobeta_orbit_patch  # noqa: E402
from lib.nospace_orbit_patch import install_nospace_orbit_patch  # noqa: E402
from lib.repro_seed import patch_parse_args  # noqa: E402
from lib.require_gpu_patch import install_require_gpu_patch  # noqa: E402
from lib.skip_stage3_patch import install_skip_stage3_patch  # noqa: E402

install_nospace_orbit_patch()

import stage2.oastid_full_huber_freezesem as F  # noqa: E402
from lib.sage_nosubgraph_patch import install_stage2_fullgraph  # noqa: E402

H = F.H
install_nobeta_orbit_patch(H)
install_skip_stage3_patch(H)
patch_parse_args(H)
install_require_gpu_patch(H)

_orig_parse = H.parse_args


def _parse_args(*args, **kwargs):
    parsed = _orig_parse(*args, **kwargs)
    parsed.no_beta_net = int(_KNOWN.no_beta_net)
    parsed.skip_stage3 = int(_KNOWN.skip_stage3)
    parsed.spatial_avwgcn = int(_KNOWN.spatial_avwgcn)
    return parsed


H.parse_args = _parse_args

_orig_erm = H.meta_train


def _meta_train_erm_freezesem(args, model, *rest, **kwargs):
    handles = F._install_sem_proj_outer_freeze(model)
    try:
        return _orig_erm(args, model, *rest, **kwargs)
    finally:
        for h in handles:
            h.remove()


H.meta_train = _meta_train_erm_freezesem
install_stage2_fullgraph(H)

if __name__ == '__main__':
    print(
        '[egbeta_limnbr] entry: Huber+freezesem+limnbr subgraph ERM; '
        f'no_beta_net={int(_KNOWN.no_beta_net)} skip_stage3={int(_KNOWN.skip_stage3)} '
        f'spatial_avwgcn={int(_KNOWN.spatial_avwgcn)} (Cross via eval_ckpt only)',
        flush=True,
    )
    H.main()
