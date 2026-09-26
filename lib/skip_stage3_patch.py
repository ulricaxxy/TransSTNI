"""Skip inline Stage3 cross at end of ``main()``; use standalone ``eval_ckpt`` instead."""
from __future__ import annotations

from typing import Any


def _skip_stage3(args: Any) -> bool:
    if not bool(int(getattr(args, 'skip_stage3', 0))):
        return False
    # Standalone eval_ckpt jobs must still run Cross.
    if getattr(args, 'eval_ckpt', None):
        return False
    return True


def install_skip_stage3_patch(huber_module: Any) -> None:
    if getattr(huber_module, '_skip_stage3_patched', False):
        return

    base = getattr(huber_module, 'base', huber_module)
    _orig = getattr(base, 'parse_target_list', huber_module.parse_target_list)

    def parse_target_list_skip(args):
        if _skip_stage3(args):
            print(
                '[skip_stage3] inline Stage3 cross disabled; '
                'run eval_ckpt on best.pt separately',
                flush=True,
            )
            return []
        return _orig(args)

    base.parse_target_list = parse_target_list_skip
    huber_module.parse_target_list = parse_target_list_skip
    huber_module._skip_stage3_patched = True
    print('[skip_stage3] patched parse_target_list', flush=True)
