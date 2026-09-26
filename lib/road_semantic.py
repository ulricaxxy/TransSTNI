"""Zero-shot road-network text + official Qwen3-Embedding-0.6B encoding.

Each LargeST meta row becomes a short English sentence. ``texts[i]`` is
aligned 1-1 with meta CSV row i (same order as npz / adjacency nodes).

Text keeps the user-specified fields and only adds transferable extras:
  Highway traffic detector.
  Route class {Interstate|US highway|state route|...}.   (Fwy prefix only)
  Travel direction {northbound|...}.
  {N} lanes.
  Detector type {Mainline|...}.                          (CSV Type, verbatim)
  ... zero-shot extras (capacity band, access class) ...
  Do not use city, county, district, route number, or geographic identity.

Dropped on purpose (they leak source-city identity):
  County, District, Lat, Lng, sensor ID, route number (I5 / US101 / SR4).

Encoding follows the official Qwen3-Embedding transformers recipe
(README): left padding, get_detailed_instruct, last_token_pool, L2 norm,
max_length=8192. The 0.6B checkpoint is loaded from the local folder.
"""
from __future__ import annotations

import hashlib
import json
import os
import re

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_QWEN_PATH = os.path.join(PROJECT_ROOT, 'Qwen3-Embedding-0.6B')
META_DIR = os.path.join(PROJECT_ROOT, 'data', 'largest_meta')
CACHE_DIR = os.path.join(META_DIR, '.qwen_cache')
TEMPLATE_VERSION = 'road_text_v2'
QWEN_MAX_LENGTH = 8192
META_CSV = {
    'largest_sd': 'sd_meta.csv',
    'largest_gba': 'gba_meta.csv',
    'largest_gla': 'gla_meta.csv',
    'largest_ca': 'ca_meta.csv',
}
ROUTE_CLASS = {
    'I': 'Interstate',
    'US': 'US highway',
    'SR': 'state route',
    'CA': 'state route',
}
DIRECTION = {
    'N': 'northbound',
    'S': 'southbound',
    'E': 'eastbound',
    'W': 'westbound',
    'NORTH': 'northbound',
    'SOUTH': 'southbound',
    'EAST': 'eastbound',
    'WEST': 'westbound',
}
DETECTOR_TYPE = {
    'mainline': 'Mainline',
    'on ramp': 'On-ramp',
    'on-ramp': 'On-ramp',
    'off ramp': 'Off-ramp',
    'off-ramp': 'Off-ramp',
    'hov': 'HOV',
    'hov out': 'HOV outbound',
    'hov-out': 'HOV outbound',
    'fwy-fwy': 'freeway connector',
    'fwy fwy': 'freeway connector',
    'ff': 'freeway connector',
    'collect/dist': 'collector-distributor',
    'collective': 'collector-distributor',
    'cd': 'collector-distributor',
    'ramp': 'Ramp',
}
QWEN_TASK = (
    'Represent this highway traffic detector using only transferable functional '
    'attributes (route class, travel direction, lane count, detector type). '
    'Ignore city, county, district, route number, and geographic identity.'
)
_FWY_PREFIX_RE = re.compile('^([A-Za-z]+)')


def get_detailed_instruct(task_description: str, query: str) -> str:
    """Official Qwen3-Embedding instruct format (verbatim from the model README)."""
    return f'Instruct: {task_description}\nQuery:{query}'


def _route_class(fwy: str) -> str:
    m = _FWY_PREFIX_RE.match(str(fwy).strip())
    if not m:
        return 'other highway'
    return ROUTE_CLASS.get(m.group(1).upper(), 'other highway')


def _direction(row) -> str:
    d = str(row.get('Direction', '')).strip().upper()
    if d in DIRECTION:
        return DIRECTION[d]
    fwy = str(row.get('Fwy', ''))
    if '-' in fwy:
        suf = fwy.rsplit('-', 1)[-1].strip().upper()
        if suf in DIRECTION:
            return DIRECTION[suf]
    return 'unknown'


def _lane_band(n: int) -> str:
    if n <= 2:
        return 'narrow capacity'
    if n <= 4:
        return 'standard capacity'
    return 'wide capacity'


def _detector_type(raw) -> str:
    """Keep the CSV Type token; only normalize known PeMS aliases."""
    raw_s = (
        ''
        if raw is None or (isinstance(raw, float) and np.isnan(raw))
        else str(raw).strip()
    )
    if not raw_s:
        return 'unknown'
    mapped = DETECTOR_TYPE.get(raw_s.lower())
    return mapped if mapped is not None else raw_s


def _access_class(det: str) -> str:
    d = det.lower()
    if d == 'mainline':
        return 'Limited-access through-traffic sensor.'
    if 'on-ramp' in d or d == 'on ramp':
        return 'On-ramp merge sensor.'
    if 'off-ramp' in d or d == 'off ramp':
        return 'Off-ramp diverge sensor.'
    if 'hov' in d:
        return 'High-occupancy-lane sensor.'
    if 'connector' in d or 'fwy' in d:
        return 'Freeway-to-freeway connector sensor.'
    if 'collector' in d:
        return 'Collector-distributor sensor.'
    return 'Limited-access highway sensor.'


def row_to_text(row) -> str:
    """User template (unmerged) + transferable extras. No geo / route number."""
    lanes = int(row['Lanes'])
    det = _detector_type(row.get('Type', 'Mainline'))
    parts = [
        'Highway traffic detector.',
        f"Route class {_route_class(row.get('Fwy', ''))}.",
        f'Travel direction {_direction(row)}.',
        f'{lanes} lanes.',
        f'Detector type {det}.',
        f'Capacity band: {_lane_band(lanes)}.',
        _access_class(det),
        'Do not use city, county, district, route number, or geographic identity.',
    ]
    return ' '.join(parts)


def load_meta_csv(dataset: str) -> pd.DataFrame:
    if dataset not in META_CSV:
        raise KeyError(f'no road meta csv for dataset={dataset}')
    path = os.path.join(META_DIR, META_CSV[dataset])
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def build_texts(dataset: str, expected_n: int | None = None) -> list[str]:
    df = load_meta_csv(dataset)
    if expected_n is not None and len(df) != int(expected_n):
        raise ValueError(
            f'{dataset} meta has {len(df)} rows, expected {expected_n} nodes'
        )
    texts = [row_to_text(row) for _, row in df.iterrows()]
    if len(texts) != len(df):
        raise RuntimeError(f'texts {len(texts)} != meta rows {len(df)}')
    return texts


def dump_texts_aligned(
    dataset: str, texts: list[str], out_path: str | None = None
) -> str:
    """Write the full texts[i] list (one string per meta row) for inspection."""
    out_path = out_path or os.path.join(META_DIR, f'{dataset}_road_texts.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)
    return out_path


def _texts_fingerprint(texts: list[str]) -> str:
    h = hashlib.sha256()
    h.update(TEMPLATE_VERSION.encode())
    h.update(QWEN_TASK.encode())
    h.update(str(QWEN_MAX_LENGTH).encode())
    for t in texts:
        h.update(b'\n')
        h.update(t.encode())
    return h.hexdigest()[:16]


def _cache_paths(dataset: str, fp: str) -> tuple[str, str, str]:
    os.makedirs(CACHE_DIR, exist_ok=True)
    stem = f'{dataset}_{TEMPLATE_VERSION}_{fp}'
    return (
        os.path.join(CACHE_DIR, stem + '.npy'),
        os.path.join(CACHE_DIR, stem + '.json'),
        os.path.join(CACHE_DIR, stem + '_texts.json'),
    )


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor):
    """Official Qwen3-Embedding last-token pooling (left or right pad)."""
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    seq_len = attention_mask.sum(dim=1) - 1
    bsz = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(bsz, device=last_hidden_states.device), seq_len
    ]


def encode_texts_qwen(
    texts: list[str], model_path: str, device, batch_size: int = 64
):
    """Official transformers encode (Qwen3-Embedding README, not a shortened path).

    * left-pad tokenizer
    * get_detailed_instruct(task, text) on every node (same protocol for all cities)
    * last_token_pool
    * L2 normalize
    * max_length = 8192
    * checkpoint dtype (bfloat16) when running on CUDA
    """
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    if not os.path.isdir(model_path):
        raise FileNotFoundError(f'Qwen embedding dir not found: {model_path}')
    if not os.path.isfile(os.path.join(model_path, 'model.safetensors')):
        raise FileNotFoundError(
            f'Qwen weights missing: {model_path}/model.safetensors'
        )
    device = torch.device(device)
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(
        model_path, padding_side='left', trust_remote_code=True
    )
    dtype = getattr(cfg, 'torch_dtype', None) or torch.bfloat16
    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=dtype
    )
    model.eval().to(device)

    chunks = []
    bs = int(batch_size)
    for i in range(0, len(texts), bs):
        raw = texts[i:i + bs]
        batch = [get_detailed_instruct(QWEN_TASK, t) for t in raw]
        enc = tok(
            batch,
            padding=True,
            truncation=True,
            max_length=QWEN_MAX_LENGTH,
            return_tensors='pt',
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        hidden = model(**enc).last_hidden_state
        pooled = last_token_pool(hidden, enc['attention_mask'])
        pooled = F.normalize(pooled.float(), p=2, dim=-1)
        chunks.append(pooled.cpu())
        print(
            f'[qwen-road] encoded {min(i + len(raw), len(texts))}/{len(texts)}',
            flush=True,
        )

    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    emb = torch.cat(chunks, dim=0).numpy().astype(np.float32)
    if emb.shape[0] != len(texts):
        raise RuntimeError(f'Qwen rows {emb.shape[0]} != texts {len(texts)}')
    if int(emb.shape[1]) != 1024:
        raise RuntimeError(
            f'Qwen3-Embedding-0.6B dim should be 1024, got {emb.shape[1]}'
        )
    return emb


def load_or_encode_road_sem(
    dataset: str,
    device,
    model_path: str | None = None,
    expected_n: int | None = None,
    batch_size: int = 64,
):
    """Return L2-normalized Qwen embeddings [N, 1024] on ``device``.

    Always materializes the full ``texts[i]`` list (aligned with meta rows).
    Encodes once and caches under data/largest_meta/.qwen_cache/.
    """
    model_path = model_path or DEFAULT_QWEN_PATH
    texts = build_texts(dataset, expected_n=expected_n)
    aligned = dump_texts_aligned(dataset, texts)
    print(
        f'[qwen-road] wrote aligned texts[i] -> {aligned} N={len(texts)}',
        flush=True,
    )
    print(f'[qwen-road] example[0]: {texts[0]}', flush=True)

    fp = _texts_fingerprint(texts)
    npy_path, json_path, texts_path = _cache_paths(dataset, fp)
    if os.path.isfile(npy_path):
        emb = np.load(npy_path)
        print(f'[qwen-road] cache hit {npy_path} shape={emb.shape}', flush=True)
    else:
        print(
            f'[qwen-road] encoding {dataset} N={len(texts)} with {model_path} '
            f'(official last_token_pool, max_length={QWEN_MAX_LENGTH})',
            flush=True,
        )
        emb = encode_texts_qwen(
            texts, model_path, device, batch_size=batch_size
        )
        np.save(npy_path, emb)
        with open(texts_path, 'w') as f:
            json.dump(texts, f, ensure_ascii=False, indent=2)
        with open(json_path, 'w') as f:
            json.dump(
                {
                    'dataset': dataset,
                    'template': TEMPLATE_VERSION,
                    'fingerprint': fp,
                    'n': len(texts),
                    'dim': int(emb.shape[1]),
                    'qwen_max_length': QWEN_MAX_LENGTH,
                    'qwen_task': QWEN_TASK,
                    'texts_path': texts_path,
                    'aligned_texts_path': aligned,
                },
                f,
                indent=2,
            )
        print(
            f'[qwen-road] saved {npy_path} and full texts {texts_path}',
            flush=True,
        )

    if expected_n is not None and emb.shape[0] != int(expected_n):
        raise ValueError(
            f'{dataset} Qwen emb N={emb.shape[0]} != expected {expected_n}'
        )
    if emb.shape[0] != len(texts):
        raise ValueError(f'emb N={emb.shape[0]} != texts {len(texts)}')
    return torch.from_numpy(np.asarray(emb, dtype=np.float32)).to(device)
