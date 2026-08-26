#!/usr/bin/env python3
"""Frame-level VAD evaluation of Silero v4 / v5 / v6 and TEN-VAD on
AISHELL-5 and VoxConverse.

Protocol, model paths, and speech/non-speech labeling are documented in
eval_vad_datasets.md. Summary:

  * Silero: stock dual-rate ONNX, 512-sample windows (31.25 ms)
  * TEN-VAD: Windows DLL, 256-sample hop (16 ms), int16 PCM
  * a frame is speech if >= 50% of it overlaps a ground-truth speech interval
  * AISHELL-5 near (DA*) and far (DX01-04) TextGrids; VoxConverse RTTM union

Usage:
    python eval_vad_datasets.py
    python eval_vad_datasets.py --models ten
    python eval_vad_datasets.py --limit 4 --workers 4
"""
from __future__ import annotations

import argparse
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
SR = 16000
WINDOW = 512  # Silero hop; TEN uses TEN_HOP
FRAME_SEC = WINDOW / SR  # 0.03125
CTX_V5 = 64
TEN_HOP = 256  # 16 ms at 16 kHz
TEN_ROOT = Path(r'D:\ten-vad')
TEN_DLL = TEN_ROOT / 'lib' / 'Windows' / 'x64' / 'ten_vad.dll'

MODELS = {
    'v4': REPO / 'examples' / 'openvino' / 'models' / 'v4_silero_vad.onnx',
    'v5': REPO / 'examples' / 'openvino' / 'models' / 'v5_silero_vad.onnx',
    'v6': REPO / 'src' / 'silero_vad' / 'data' / 'silero_vad.onnx',
    'ten': TEN_DLL,
}

MODEL_WINDOW = {
    'v4': WINDOW,
    'v5': WINDOW,
    'v6': WINDOW,
    'ten': TEN_HOP,
}

AISHELL_ROOT = Path(r'D:\AISHELL-5-Data')
VOX_ROOT = Path(r'D:\voxconverse')
FARFIELD_STEMS = {'DX01C01', 'DX02C01', 'DX03C01', 'DX04C01'}
NEARFIELD_PREFIX = 'DA'  # headset mics, e.g. DA01 / DA03


# ---------------------------------------------------------------------------
# Annotations
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
    """Praat long TextGrid: any non-empty interval text is speech."""
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
    """Majority overlap (>= 50% of the real frame duration)."""
    labels = np.zeros(n_frames, dtype=np.uint8)
    dur = np.full(n_frames, frame_sec, dtype=np.float64)
    last_real = n_samples / SR
    last_i = n_frames - 1
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
    kind: str  # 'textgrid' | 'rttm'


def list_aishell(splits: Sequence[str]) -> List[Item]:
    """Far-field DX01-04 (in-car) and near-field DA* (headset), separately."""
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

_SESSION = None
_MODEL_KIND = None
_WINDOW = WINDOW


def _make_session(model_path: str) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.inter_op_num_threads = 1
    so.intra_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        model_path, so, providers=['CPUExecutionProvider'])


def _load_ten_vad():
    include = str(TEN_ROOT / 'include')
    if include not in sys.path:
        sys.path.insert(0, include)
    dll_dir = str(TEN_DLL.parent)
    if hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(dll_dir)
    from ten_vad import TenVad  # noqa: WPS433
    return TenVad


def _init_worker(model_path: str, model_kind: str, window: int) -> None:
    global _SESSION, _MODEL_KIND, _WINDOW
    _MODEL_KIND = model_kind
    _WINDOW = int(window)
    if model_kind == 'ten':
        _SESSION = None
        _load_ten_vad()
    else:
        _SESSION = _make_session(model_path)


def infer_probs(session: ort.InferenceSession, kind: str,
                wav: np.ndarray, window: int) -> np.ndarray:
    n = int(wav.shape[0])
    n_frames = (n + window - 1) // window
    pad = n_frames * window - n
    if pad:
        wav = np.pad(wav, (0, pad))
    sr = np.array(SR, dtype=np.int64)
    probs = np.empty(n_frames, dtype=np.float32)

    if kind == 'v4':
        h = np.zeros((2, 1, 64), np.float32)
        c = np.zeros((2, 1, 64), np.float32)
        x = np.empty((1, window), np.float32)
        for i in range(n_frames):
            x[0] = wav[i * window:(i + 1) * window]
            out, h, c = session.run(
                ['output', 'hn', 'cn'],
                {'input': x, 'sr': sr, 'h': h, 'c': c})
            probs[i] = out[0, 0]
        return probs

    state = np.zeros((2, 1, 128), np.float32)
    x = np.zeros((1, CTX_V5 + window), np.float32)
    for i in range(n_frames):
        x[0, CTX_V5:] = wav[i * window:(i + 1) * window]
        out, state = session.run(
            ['output', 'stateN'],
            {'input': x, 'state': state, 'sr': sr})
        x[0, :CTX_V5] = x[0, -CTX_V5:]
        probs[i] = out[0, 0]
    return probs


def infer_ten(pcm16: np.ndarray, hop: int) -> np.ndarray:
    """Stream TEN-VAD; new handle per file so LSTM/history cannot leak."""
    TenVad = _load_ten_vad()
    n = int(pcm16.shape[0])
    n_frames = (n + hop - 1) // hop
    pad = n_frames * hop - n
    if pad:
        pcm16 = np.pad(pcm16, (0, pad))
    pcm16 = np.ascontiguousarray(pcm16, dtype=np.int16)
    vad = TenVad(hop_size=hop, threshold=0.5)
    try:
        probs = np.empty(n_frames, dtype=np.float32)
        for i in range(n_frames):
            chunk = pcm16[i * hop:(i + 1) * hop]
            prob, _flag = vad.process(chunk)
            probs[i] = prob
        return probs
    finally:
        del vad


def _process_item(item: Item) -> dict:
    window = _WINDOW
    if _MODEL_KIND == 'ten':
        wav, sr = sf.read(item.wav, dtype='int16', always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1).astype(np.int16)
    else:
        wav, sr = sf.read(item.wav, dtype='float32', always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
    if sr != SR:
        raise RuntimeError(f'{item.wav}: expected {SR} Hz, got {sr}')
    n_samples = int(wav.shape[0])
    n_frames = (n_samples + window - 1) // window
    if item.kind == 'textgrid':
        intervals = parse_textgrid(Path(item.ann))
    else:
        intervals = parse_rttm(Path(item.ann))
    labels = intervals_to_frames(
        intervals, n_frames, n_samples, window / SR)
    t0 = time.perf_counter()
    if _MODEL_KIND == 'ten':
        probs = infer_ten(wav, window)
    else:
        probs = infer_probs(_SESSION, _MODEL_KIND, wav, window)
    infer_sec = time.perf_counter() - t0
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
    return float(np.trapz(tps / n_pos, fps / n_neg))


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
    if x != x:  # NaN
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


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_model(model_name: str, items: List[Item], workers: int,
              limit: Optional[int]) -> List[dict]:
    path = MODELS[model_name]
    window = MODEL_WINDOW[model_name]
    if not path.is_file():
        raise FileNotFoundError(path)
    subset = items[:limit] if limit else items
    ctx = mp.get_context('spawn')
    n_workers = max(1, min(workers, len(subset)))
    print(f'\n=== {model_name}  {path.name}  hop={window}  files={len(subset)}  '
          f'workers={n_workers} ===')
    t0 = time.perf_counter()
    rows: List[dict] = []
    with ctx.Pool(n_workers, initializer=_init_worker,
                  initargs=(str(path), model_name, window)) as pool:
        for r in tqdm(pool.imap_unordered(_process_item, subset, chunksize=1),
                      total=len(subset), unit='utt'):
            rows.append(r)
    print(f'{model_name}: wall {time.perf_counter() - t0:.1f}s')
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--models', nargs='+', default=['v4', 'v5', 'v6'],
                   choices=list(MODELS))
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

    payload = {
        'threshold': args.threshold,
        'window_samples': {n: MODEL_WINDOW[n] for n in args.models},
        'sample_rate': SR,
        'models': {},
    }
    keys_main = ('files', 'hours', 'roc_auc', 'accuracy', 'f1',
                 'precision', 'recall', 'far', 'mr', 'rtf')

    for model_name in args.models:
        all_rows: List[dict] = []
        model_out = {'by_split': {}, 'by_dataset': {}, 'overall': {}}
        for ds, ds_items in by_ds.items():
            rows = run_model(model_name, ds_items, args.workers, args.limit)
            all_rows.extend(rows)
            # per split
            splits = sorted({r['split'] for r in rows})
            for sp in splits:
                sub = [r for r in rows if r['split'] == sp]
                key = f'{ds}/{sp}'
                model_out['by_split'][key] = summarize(sub, args.threshold)
            model_out['by_dataset'][ds] = summarize(rows, args.threshold)
        model_out['overall'] = summarize(all_rows, args.threshold)

        # best-F1 threshold on each dataset's first split (dev), applied to test
        for ds, rows_ds in (
            (ds, [r for r in all_rows if r['dataset'] == ds])
            for ds in by_ds
        ):
            val_split = 'dev' if ds == 'voxconverse' else 'dev'
            val = [r for r in rows_ds if r['split'] == val_split]
            test = [r for r in rows_ds if r['split'] != val_split]
            if val:
                y = np.concatenate([r['labels'] for r in val])
                s = np.concatenate([r['probs'] for r in val])
                t, best = best_threshold(y, s)
                model_out['by_dataset'][ds]['best_f1_on_dev'] = best
                if test:
                    yt = np.concatenate([r['labels'] for r in test])
                    st = np.concatenate([r['probs'] for r in test])
                    tuned = metrics_from_confusion(*confusion(yt, st >= t))
                    tuned['threshold'] = t
                    tuned['roc_auc'] = roc_auc(yt, st)
                    model_out['by_dataset'][ds]['test_at_dev_best_t'] = tuned

        payload['models'][model_name] = {
            'window_samples': MODEL_WINDOW[model_name],
            'artifact': str(MODELS[model_name]),
            'by_split': {k: jsonable(v) for k, v in model_out['by_split'].items()},
            'by_dataset': {k: jsonable(v) for k, v in model_out['by_dataset'].items()},
            'overall': jsonable(model_out['overall']),
        }

        print_table(f'{model_name} @ t={args.threshold}',
                    model_out['by_split'], keys_main)
        print_table(f'{model_name} datasets',
                    model_out['by_dataset'], keys_main)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.is_file():
        prev = json.loads(args.out.read_text(encoding='utf-8'))
        prev_models = prev.get('models', {})
        prev_ws = prev.get('window_samples', WINDOW)
        if isinstance(prev_ws, dict):
            ws = dict(prev_ws)
        else:
            ws = {k: int(prev_ws) for k in prev_models}
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
    # Windows spawn + tqdm: avoid extra blank lines from buffering
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    raise SystemExit(main())
