#!/usr/bin/env python3
"""Frame-level VAD evaluation of FireRed Stream-VAD (ONNX with cache)
on AISHELL-5 and VoxConverse.

Same dataset scope, speech/non-speech labeling, and metrics as
eval_vad_datasets.py / eval_vad_datasets.md. Differences:

  * model: fireredvad_stream_vad_with_cache.onnx
  * front-end: Kaldi fbank 80-dim, 25 ms window / 10 ms hop, snip_edges=True, CMVN
  * hop for labels: 160 samples (10 ms); leftover tail after last fbank frame
    is unlabeled (official snip_edges drops an incomplete 25 ms window)
  * raw ONNX probs only — no StreamVadPostprocessor
  * run in conda env fireredvad

Usage (conda env fireredvad):
    python examples/eval_fireredvad_datasets.py
    python examples/eval_fireredvad_datasets.py --limit 2 --workers 4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnxruntime as ort
import soundfile as sf
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
FIRERED_ROOT = Path(r'D:\FireRedVAD')
if str(FIRERED_ROOT) not in sys.path:
    sys.path.insert(0, str(FIRERED_ROOT))

from fireredvad.core.audio_feat import AudioFeat  # noqa: E402

SR = 16000
HOP = 160  # 10 ms
FRAME_SEC = HOP / SR
ONNX_PATH = (FIRERED_ROOT / 'pretrained_models' / 'onnx_models'
             / 'fireredvad_stream_vad_with_cache.onnx')
CMVN_PATH = (FIRERED_ROOT / 'pretrained_models' / 'onnx_models' / 'cmvn.ark')

NUM_CACHES = 8
CACHE_P = 128
CACHE_LEN = 19
CACHE_SHAPE = (NUM_CACHES, 1, CACHE_P, CACHE_LEN)

AISHELL_ROOT = Path(r'D:\AISHELL-5-Data')
VOX_ROOT = Path(r'D:\voxconverse')
FARFIELD_STEMS = {'DX01C01', 'DX02C01', 'DX03C01', 'DX04C01'}
NEARFIELD_PREFIX = 'DA'

_SESSION = None
_AUDIO_FEAT = None


# ---------------------------------------------------------------------------
# Annotations (same rules as eval_vad_datasets.py)
# ---------------------------------------------------------------------------

def merge_intervals(segs: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not segs:
        return []
    segs = sorted((float(s), float(e)) for s, e in segs if e > s)
    out = [segs[0]]
    for s, e in segs[1:]:
        if s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def parse_textgrid(path: Path) -> List[Tuple[float, float]]:
    raw = path.read_text(encoding='utf-8', errors='replace')
    segs: List[Tuple[float, float]] = []
    pat = re.compile(
        r'intervals\s*\[\d+\]:\s*'
        r'xmin\s*=\s*([0-9.]+)\s*'
        r'xmax\s*=\s*([0-9.]+)\s*'
        r'text\s*=\s*"((?:[^"]|"")*)"',
    )
    for m in pat.finditer(raw):
        text = m.group(3).replace('""', '"').strip()
        if text:
            segs.append((float(m.group(1)), float(m.group(2))))
    return merge_intervals(segs)


def parse_rttm(path: Path) -> List[Tuple[float, float]]:
    segs: List[Tuple[float, float]] = []
    with path.open(encoding='utf-8', errors='replace') as f:
        for line in f:
            if not line.startswith('SPEAKER'):
                continue
            p = line.split()
            start, dur = float(p[3]), float(p[4])
            segs.append((start, start + dur))
    return merge_intervals(segs)


def intervals_to_frames(intervals: Sequence[Tuple[float, float]],
                        n_frames: int, n_samples: int,
                        frame_sec: float) -> np.ndarray:
    """Majority overlap (>= 50% of the hop interval that actually exists).

    Each frame i is the hop [i * hop, (i+1) * hop). FireRed fbank uses
    snip_edges=True, so hops usually all fit inside the wav; the last hop is
    clipped only if it would extend past the audio. Unframed tail samples
    after the last fbank frame have no label and no model output.
    """
    labels = np.zeros(n_frames, dtype=np.uint8)
    if n_frames == 0:
        return labels
    dur = np.full(n_frames, frame_sec, dtype=np.float64)
    last_real = n_samples / SR
    last_i = n_frames - 1
    hop_end = (last_i + 1) * frame_sec
    if hop_end > last_real:
        dur[last_i] = max(last_real - last_i * frame_sec, 1e-9)
    for s, e in intervals:
        i0 = max(int(s / frame_sec), 0)
        i1 = min(int(np.ceil(e / frame_sec)), n_frames)
        for i in range(i0, i1):
            t0 = i * frame_sec
            t1 = t0 + dur[i]
            ov = min(t1, e) - max(t0, s)
            if ov >= 0.5 * dur[i]:
                labels[i] = 1
    return labels


# ---------------------------------------------------------------------------
# Dataset index
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Item:
    dataset: str
    split: str
    utt: str
    wav: str
    ann: str
    kind: str


def list_aishell(splits: Sequence[str]) -> List[Item]:
    items: List[Item] = []
    for split in splits:
        root = AISHELL_ROOT / split
        if not root.is_dir():
            continue
        for wav in sorted(root.rglob('*.wav')):
            tg = wav.with_suffix('.TextGrid')
            if not tg.is_file():
                continue
            if wav.stem in FARFIELD_STEMS:
                ds = 'aishell5_far'
            elif wav.stem.startswith(NEARFIELD_PREFIX):
                ds = 'aishell5_near'
            else:
                continue
            session = wav.parent.name
            items.append(Item(
                dataset=ds,
                split=split.lower(),
                utt=f'{split}/{session}/{wav.stem}',
                wav=str(wav),
                ann=str(tg),
                kind='textgrid',
            ))
    return items


def list_voxconverse(splits: Sequence[str]) -> List[Item]:
    wav_dirs = {
        'dev': VOX_ROOT / 'voxconverse_dev_wav',
        'test': VOX_ROOT / 'voxconverse_test_wav',
    }
    items: List[Item] = []
    for split in splits:
        rttm_dir = VOX_ROOT / split
        wav_root = wav_dirs[split]
        wavs = {p.stem: p for p in wav_root.rglob('*.wav')
                if not p.name.startswith('._')}
        for rttm in sorted(rttm_dir.glob('*.rttm')):
            wav = wavs.get(rttm.stem)
            if wav is None:
                continue
            items.append(Item(
                dataset='voxconverse',
                split=split,
                utt=rttm.stem,
                wav=str(wav),
                ann=str(rttm),
                kind='rttm',
            ))
    return items


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _make_session(model_path: str) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.inter_op_num_threads = 1
    so.intra_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        model_path, so, providers=['CPUExecutionProvider'])


def _init_worker(onnx_path: str, cmvn_path: str) -> None:
    global _SESSION, _AUDIO_FEAT
    _SESSION = _make_session(onnx_path)
    _AUDIO_FEAT = AudioFeat(cmvn_path)


def infer_onnx_streaming(session: ort.InferenceSession,
                         feat: np.ndarray) -> np.ndarray:
    """Frame-by-frame ONNX with caches_in / caches_out. New zeros each call."""
    n = int(feat.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=np.float32)
    caches = np.zeros(CACHE_SHAPE, dtype=np.float32)
    probs = np.empty(n, dtype=np.float32)
    for i in range(n):
        frame = feat[np.newaxis, i:i + 1, :]
        outputs = session.run(None, {'feat': frame, 'caches_in': caches})
        probs[i] = float(np.squeeze(outputs[0]))
        caches = np.asarray(outputs[1], dtype=np.float32)
        if caches.shape != CACHE_SHAPE:
            if caches.ndim == 4 and caches.shape[:3] == CACHE_SHAPE[:3]:
                caches = caches[..., -CACHE_LEN:]
            if caches.shape != CACHE_SHAPE:
                raise RuntimeError(f'Unexpected caches_out shape: {caches.shape}')
    return probs


def _process_item(item: Item) -> dict:
    wav, sr = sf.read(item.wav, dtype='int16', always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1).astype(np.int16)
    if sr != SR:
        raise RuntimeError(f'{item.wav}: expected {SR} Hz, got {sr}')
    n_samples = int(wav.shape[0])
    if item.kind == 'textgrid':
        intervals = parse_textgrid(Path(item.ann))
    else:
        intervals = parse_rttm(Path(item.ann))

    t0 = time.perf_counter()
    feat_t, _dur = _AUDIO_FEAT.extract((wav, sr))
    feat = np.ascontiguousarray(feat_t.numpy(), dtype=np.float32)
    probs = infer_onnx_streaming(_SESSION, feat)
    infer_sec = time.perf_counter() - t0

    n_frames = int(probs.shape[0])
    labels = intervals_to_frames(intervals, n_frames, n_samples, FRAME_SEC)
    duration = n_samples / SR
    return {
        'dataset': item.dataset,
        'split': item.split,
        'utt': item.utt,
        'n_frames': n_frames,
        'duration': duration,
        'infer_sec': infer_sec,
        'probs': probs,
        'labels': labels,
        'speech_sec': float(sum(e - s for s, e in intervals)),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.bool_)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int(y_true.sum())
    n_neg = int(y_true.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    order = np.argsort(y_score, kind='mergesort')[::-1]
    y_true = y_true[order]
    y_score = y_score[order]
    distinct = np.where(np.diff(y_score))[0]
    idxs = np.r_[distinct, y_true.size - 1]
    tps = np.cumsum(y_true)[idxs]
    fps = 1 + idxs - tps
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    trapz = getattr(np, 'trapezoid', np.trapz)
    return float(trapz(tps / n_pos, fps / n_neg))


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    y_true = np.asarray(y_true, dtype=np.bool_)
    y_pred = np.asarray(y_pred, dtype=np.bool_)
    tp = int(np.count_nonzero(y_true & y_pred))
    fp = int(np.count_nonzero(~y_true & y_pred))
    fn = int(np.count_nonzero(y_true & ~y_pred))
    tn = int(np.count_nonzero(~y_true & ~y_pred))
    return tp, fp, fn, tn


def metrics_from_confusion(tp: int, fp: int, fn: int, tn: int) -> dict:
    n = tp + fp + fn + tn
    acc = (tp + tn) / n if n else float('nan')
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    far = fp / (fp + tn) if (fp + tn) else 0.0
    mr = fn / (fn + tp) if (fn + tp) else 0.0
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'far': far,
        'mr': mr,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
    }


def best_threshold(y_true: np.ndarray, y_score: np.ndarray,
                   grid: Optional[np.ndarray] = None) -> Tuple[float, dict]:
    if grid is None:
        grid = np.round(np.linspace(0.05, 0.95, 19), 2)
    best_t, best = 0.5, None
    for t in grid:
        m = metrics_from_confusion(*confusion(y_true, y_score >= t))
        m['threshold'] = float(t)
        if best is None or m['f1'] > best['f1']:
            best_t, best = float(t), m
    return best_t, best


def summarize(rows: List[dict], threshold: float = 0.5) -> dict:
    rows = [r for r in rows if r['n_frames'] > 0]
    labels = np.concatenate([r['labels'] for r in rows])
    probs = np.concatenate([r['probs'] for r in rows])
    duration = float(sum(r['duration'] for r in rows))
    infer_sec = float(sum(r['infer_sec'] for r in rows))
    speech_sec = float(sum(r['speech_sec'] for r in rows))
    m = metrics_from_confusion(*confusion(labels, probs >= threshold))
    m.update({
        'files': len(rows),
        'hours': duration / 3600.0,
        'speech_ratio': speech_sec / duration if duration else 0.0,
        'roc_auc': roc_auc(labels, probs),
        'threshold': threshold,
        'rtf': infer_sec / duration if duration else float('nan'),
        'infer_sec': infer_sec,
        'n_frames': int(labels.size),
    })
    return m


def fmt(x: float, digits: int = 3) -> str:
    if x != x:
        return '  n/a'
    return f'{x:.{digits}f}'


def print_table(title: str, results: Dict[str, dict],
                keys: Sequence[str]) -> None:
    header = f'{title:<28}' + ''.join(f'{k:>10}' for k in keys)
    print('\n' + header)
    print('-' * len(header))
    for name, row in results.items():
        line = f'{name:<28}'
        for k in keys:
            v = row.get(k, float('nan'))
            if k in ('hours', 'rtf', 'threshold'):
                line += f'{fmt(v, 3):>10}'
            elif k == 'files':
                line += f'{int(v):>10d}'
            else:
                line += f'{fmt(v, 3):>10}'
        print(line)


def jsonable(m: dict) -> dict:
    out = {}
    for k, v in m.items():
        if isinstance(v, (np.floating, float)):
            out[k] = None if v != v else float(v)
        elif isinstance(v, (np.integer, int)):
            out[k] = int(v)
        else:
            out[k] = v
    return out


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_items(items: List[Item], workers: int,
              limit: Optional[int]) -> List[dict]:
    if not ONNX_PATH.is_file():
        raise FileNotFoundError(ONNX_PATH)
    if not CMVN_PATH.is_file():
        raise FileNotFoundError(CMVN_PATH)
    subset = items[:limit] if limit else items
    ctx = mp.get_context('spawn')
    n_workers = max(1, min(workers, len(subset)))
    print(f'\n=== firered  {ONNX_PATH.name}  hop={HOP}  files={len(subset)}  '
          f'workers={n_workers} ===')
    t0 = time.perf_counter()
    rows: List[dict] = []
    with ctx.Pool(n_workers, initializer=_init_worker,
                  initargs=(str(ONNX_PATH), str(CMVN_PATH))) as pool:
        for r in tqdm(pool.imap_unordered(_process_item, subset, chunksize=1),
                      total=len(subset), unit='utt'):
            rows.append(r)
    print(f'firered: wall {time.perf_counter() - t0:.1f}s')
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--aishell-splits', nargs='+', default=['Dev', 'Eval1', 'Eval2'])
    p.add_argument('--vox-splits', nargs='+', default=['dev', 'test'])
    p.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 4) - 2))
    p.add_argument('--limit', type=int, default=None,
                   help='cap files per dataset (smoke test)')
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--out', type=Path,
                   default=REPO / 'examples' / 'eval_vad_results.json')
    args = p.parse_args(argv)

    items: List[Item] = []
    items.extend(list_aishell(args.aishell_splits))
    items.extend(list_voxconverse(args.vox_splits))
    if not items:
        print('No evaluation items found.', file=sys.stderr)
        return 1

    by_ds: Dict[str, List[Item]] = {}
    for it in items:
        by_ds.setdefault(it.dataset, []).append(it)
    print('Indexed:')
    for ds, lst in by_ds.items():
        splits = {}
        for it in lst:
            splits[it.split] = splits.get(it.split, 0) + 1
        print(f'  {ds}: {len(lst)} files  {splits}')
    print(f'onnx: {ONNX_PATH}')
    print(f'cmvn: {CMVN_PATH}')

    all_rows: List[dict] = []
    model_out = {'by_split': {}, 'by_dataset': {}, 'overall': {}}
    for ds, ds_items in by_ds.items():
        rows = run_items(ds_items, args.workers, args.limit)
        all_rows.extend(rows)
        splits = sorted({r['split'] for r in rows})
        for sp in splits:
            sub = [r for r in rows if r['split'] == sp]
            key = f'{ds}/{sp}'
            model_out['by_split'][key] = summarize(sub, args.threshold)
        model_out['by_dataset'][ds] = summarize(rows, args.threshold)
    model_out['overall'] = summarize(all_rows, args.threshold)

    for ds, rows_ds in (
        (ds, [r for r in all_rows if r['dataset'] == ds])
        for ds in by_ds
    ):
        val = [r for r in rows_ds if r['split'] == 'dev']
        test = [r for r in rows_ds if r['split'] != 'dev']
        if val:
            y = np.concatenate([r['labels'] for r in val if r['n_frames']])
            s = np.concatenate([r['probs'] for r in val if r['n_frames']])
            t, best = best_threshold(y, s)
            model_out['by_dataset'][ds]['best_f1_on_dev'] = best
            if test:
                yt = np.concatenate([r['labels'] for r in test if r['n_frames']])
                st = np.concatenate([r['probs'] for r in test if r['n_frames']])
                tuned = metrics_from_confusion(*confusion(yt, st >= t))
                tuned['threshold'] = t
                tuned['roc_auc'] = roc_auc(yt, st)
                model_out['by_dataset'][ds]['test_at_dev_best_t'] = tuned

    firered_payload = {
        'window_samples': HOP,
        'artifact': str(ONNX_PATH),
        'cmvn': str(CMVN_PATH),
        'sha256': file_sha256(ONNX_PATH),
        'cmvn_sha256': file_sha256(CMVN_PATH),
        'by_split': {k: jsonable(v) for k, v in model_out['by_split'].items()},
        'by_dataset': {k: jsonable(v) for k, v in model_out['by_dataset'].items()},
        'overall': jsonable(model_out['overall']),
    }

    keys_main = ('files', 'hours', 'roc_auc', 'accuracy', 'f1',
                 'precision', 'recall', 'far', 'mr', 'rtf')
    print_table(f'firered @ t={args.threshold}', model_out['by_split'], keys_main)
    print_table('firered datasets', model_out['by_dataset'], keys_main)

    payload = {
        'threshold': args.threshold,
        'window_samples': {'firered': HOP},
        'sample_rate': SR,
        'models': {'firered': firered_payload},
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.is_file():
        prev = json.loads(args.out.read_text(encoding='utf-8'))
        prev_models = prev.get('models', {})
        prev_ws = prev.get('window_samples', {})
        if not isinstance(prev_ws, dict):
            prev_ws = {k: int(prev_ws) for k in prev_models}
        ws = dict(prev_ws)
        ws.update(payload['window_samples'])
        prev_models.update(payload['models'])
        payload['models'] = prev_models
        payload['window_samples'] = ws
        payload['threshold'] = prev.get('threshold', args.threshold)
        payload['sample_rate'] = prev.get('sample_rate', SR)

    print('\n' + '=' * 88)
    model_names = list(payload['models'])
    print(f'{"split":<28}' + ''.join(f'{n:>20}' for n in model_names))
    print('-' * (28 + 20 * len(model_names)))
    print('values are  AUC / Acc@0.5 / F1@0.5')
    splits = sorted({k for m in payload['models'].values()
                     for k in m.get('by_split', {})})
    for sp in splits:
        line = f'{sp:<28}'
        for n in model_names:
            r = payload['models'][n].get('by_split', {}).get(sp, {})
            cell = (f'{fmt(r.get("roc_auc", float("nan")))}/'
                    f'{fmt(r.get("accuracy", float("nan")))}/'
                    f'{fmt(r.get("f1", float("nan")))}')
            line += f'{cell:>20}'
        print(line)
    for ds in sorted({k for m in payload['models'].values()
                      for k in m.get('by_dataset', {})}):
        line = f'{ds + " (all)":<28}'
        for n in model_names:
            r = payload['models'][n].get('by_dataset', {}).get(ds, {})
            cell = (f'{fmt(r.get("roc_auc", float("nan")))}/'
                    f'{fmt(r.get("accuracy", float("nan")))}/'
                    f'{fmt(r.get("f1", float("nan")))}')
            line += f'{cell:>20}'
        print(line)

    args.out.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(f'\nWrote {args.out}')
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    raise SystemExit(main())
