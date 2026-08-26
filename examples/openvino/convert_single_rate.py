#!/usr/bin/env python3
"""Specialize a Silero VAD ONNX graph to a single sample rate.

Works on stock dual-rate exports from v4 / v5 / v6:

* inlines every ``If`` (sample-rate switch + TorchScript leftovers)
* drops the unused rate branch and the ``sr`` input
* freezes batch=1 and a fixed window length
* runs ``onnxsim.simplify`` (constant folding / dead-code / fusion)

v4 I/O differs from v5/v6:

======= ============================= =================================
Version Inputs                        Outputs
======= ============================= =================================
v4      input, sr, h, c               output, hn, cn
v5/v6   input, state, sr              output, stateN
======= ============================= =================================

v4 feeds raw window samples (no context concat). v5/v6 expect
``context + window`` already concatenated on ``input``
(64+512 at 16 kHz, 32+256 at 8 kHz).

Usage:
    python convert_single_rate.py INPUT.onnx [-o OUT.onnx] [--sr 16000]
         [--window 512] [--verify] [--no-simplify]

Requires: numpy, onnx, onnxruntime, onnxsim.
"""
from __future__ import annotations

import argparse
import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, shape_inference

# Reuse the structural transforms from the OpenVINO converter.
from convert import (  # noqa: E402
    dce,
    eliminate_identity,
    eval_cond,
    inline_if,
)


@dataclass(frozen=True)
class ModelLayout:
    """Detected I/O contract of a stock Silero ONNX export."""

    kind: str  # 'v4' | 'v5'
    input_names: Tuple[str, ...]
    output_names: Tuple[str, ...]
    state_names: Tuple[str, ...]          # graph inputs that carry RNN state
    state_out_names: Tuple[str, ...]      # matching state outputs
    state_hidden: int
    needs_context: bool                   # True => input is context+window


def _user_inputs(model: onnx.ModelProto) -> List[onnx.ValueInfoProto]:
    inits = {i.name for i in model.graph.initializer}
    return [i for i in model.graph.input if i.name not in inits]


def detect_layout(model: onnx.ModelProto) -> ModelLayout:
    names = {i.name for i in _user_inputs(model)}
    outs = {o.name for o in model.graph.output}

    if {'input', 'sr', 'h', 'c'} <= names and {'output', 'hn', 'cn'} <= outs:
        return ModelLayout(
            kind='v4',
            input_names=('input', 'h', 'c', 'sr'),
            output_names=('output', 'hn', 'cn'),
            state_names=('h', 'c'),
            state_out_names=('hn', 'cn'),
            state_hidden=64,
            needs_context=False,
        )
    if {'input', 'state', 'sr'} <= names and {'output', 'stateN'} <= outs:
        return ModelLayout(
            kind='v5',
            input_names=('input', 'state', 'sr'),
            output_names=('output', 'stateN'),
            state_names=('state',),
            state_out_names=('stateN',),
            state_hidden=128,
            needs_context=True,
        )
    raise ValueError(
        f'Unsupported Silero ONNX I/O. inputs={sorted(names)} '
        f'outputs={sorted(outs)}. Expected v4 (h/c) or v5/v6 (state).')


def window_and_input_len(layout: ModelLayout, sr: int, window: int | None
                         ) -> Tuple[int, int]:
    """Return (window_samples, onnx_input_length)."""
    if layout.kind == 'v4':
        if window is None:
            window = 512 if sr == 16000 else 256
        # Official training windows; other lengths often still run.
        valid_16k = {512, 1024, 1536}
        valid_8k = {256, 512, 768}
        allowed = valid_16k if sr == 16000 else valid_8k
        if window not in allowed:
            print(f'warning: window={window} is outside the documented '
                  f'{sr} Hz set {sorted(allowed)}; conversion may still work',
                  file=sys.stderr)
        return window, window

    # v5 / v6
    if window is None:
        window = 512 if sr == 16000 else 256
    ctx = 64 if sr == 16000 else 32
    expected = 512 if sr == 16000 else 256
    if window != expected:
        raise ValueError(
            f'v5/v6 only support fixed window {expected} at {sr} Hz '
            f'(got {window})')
    return window, window + ctx


def build_feeds(layout: ModelLayout, sr: int, input_len: int
                ) -> Dict[str, np.ndarray]:
    feeds: Dict[str, np.ndarray] = {
        'input': np.zeros((1, input_len), dtype=np.float32),
        'sr': np.array(sr, dtype=np.int64),
    }
    if layout.kind == 'v4':
        feeds['h'] = np.zeros((2, 1, layout.state_hidden), dtype=np.float32)
        feeds['c'] = np.zeros((2, 1, layout.state_hidden), dtype=np.float32)
    else:
        feeds['state'] = np.zeros((2, 1, layout.state_hidden), dtype=np.float32)
    return feeds


def freeze_shapes(model: onnx.ModelProto, layout: ModelLayout,
                  input_len: int) -> None:
    input_dims = {
        'input': [1, input_len],
    }
    if layout.kind == 'v4':
        input_dims['h'] = [2, 1, layout.state_hidden]
        input_dims['c'] = [2, 1, layout.state_hidden]
        output_dims = {
            'output': [1, 1],
            'hn': [2, 1, layout.state_hidden],
            'cn': [2, 1, layout.state_hidden],
        }
    else:
        input_dims['state'] = [2, 1, layout.state_hidden]
        output_dims = {
            'output': [1, 1],
            'stateN': [2, 1, layout.state_hidden],
        }

    for i in model.graph.input:
        if i.name not in input_dims:
            continue
        for d, v in zip(i.type.tensor_type.shape.dim, input_dims[i.name]):
            d.ClearField('dim_param')
            d.dim_value = v
    for o in model.graph.output:
        dims = output_dims[o.name]
        # Some exports leave empty shape lists; rebuild if needed.
        while len(o.type.tensor_type.shape.dim) < len(dims):
            o.type.tensor_type.shape.dim.add()
        for d, v in zip(o.type.tensor_type.shape.dim, dims):
            d.ClearField('dim_param')
            d.dim_value = v
    del model.graph.value_info[:]


def simplify_graph(model: onnx.ModelProto, layout: ModelLayout,
                   input_len: int) -> onnx.ModelProto:
    """Constant-fold and fuse with onnxsim after If inlining."""
    from onnxsim import simplify

    input_shapes = {'input': [1, input_len]}
    if layout.kind == 'v4':
        input_shapes['h'] = [2, 1, layout.state_hidden]
        input_shapes['c'] = [2, 1, layout.state_hidden]
    else:
        input_shapes['state'] = [2, 1, layout.state_hidden]

    n_before = len(model.graph.node)
    simplified, ok = simplify(
        model,
        overwrite_input_shapes=input_shapes,
        skipped_optimizers=['fuse_bn'],
    )
    if not ok:
        raise RuntimeError('onnxsim.simplify failed its numerical check')
    print(f'onnxsim: nodes {n_before} -> {len(simplified.graph.node)}')
    return simplified


def convert(src: Path, dst: Path, sr: int = 16000,
            window: int | None = None, do_simplify: bool = True) -> ModelLayout:
    model = onnx.load(str(src))
    layout = detect_layout(model)
    window, input_len = window_and_input_len(layout, sr, window)
    feeds = build_feeds(layout, sr, input_len)

    print(f'detected {layout.kind}-style ONNX')
    print(f'  keep sr={sr}, window={window}, onnx input length={input_len}')

    step = 0
    while True:
        ifs = [(i, n) for i, n in enumerate(model.graph.node)
               if n.op_type == 'If']
        if not ifs:
            break
        idx, node = ifs[0]
        cond = eval_cond(model, node.input[0], feeds)
        branch_name = 'then_branch' if cond else 'else_branch'
        branch = next(a.g for a in node.attribute if a.name == branch_name)
        print(f'[{step}] inline If {node.name!r}: cond={cond} -> {branch_name} '
              f'({len(branch.node)} nodes)')
        inline_if(model.graph, idx, branch, f'F{step}')
        step += 1

    leftover = [n.op_type for n in model.graph.node
                if n.op_type in ('If', 'Loop', 'Scan')]
    if leftover:
        raise RuntimeError(f'control-flow nodes remain: {leftover}')

    dce(model.graph)
    eliminate_identity(model.graph)
    freeze_shapes(model, layout, input_len)

    model = shape_inference.infer_shapes(model)
    if do_simplify:
        model = simplify_graph(model, layout, input_len)
        model = shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    onnx.save(model, str(dst))

    kept_in = [i.name for i in _user_inputs(model)]
    print(f'\nwrote {dst}')
    print(f'  nodes={len(model.graph.node)} If=0 '
          f'inputs={kept_in} outputs={[o.name for o in model.graph.output]}')
    if 'sr' in kept_in:
        raise RuntimeError('sr input was not eliminated')
    return layout


def _run_stock(session, layout: ModelLayout, feeds: Dict[str, np.ndarray]):
    names = [i.name for i in session.get_inputs()]
    return session.run(None, {k: feeds[k] for k in names})


def _run_converted(session, layout: ModelLayout, feeds: Dict[str, np.ndarray]):
    names = [i.name for i in session.get_inputs()]
    return session.run(None, {k: feeds[k] for k in names if k != 'sr'})


def verify(stock: Path, converted: Path, layout: ModelLayout, sr: int,
           window: int, input_len: int, chunks: int = 64) -> None:
    so = ort.SessionOptions()
    so.log_severity_level = 3
    stock_s = ort.InferenceSession(
        str(stock), so, providers=['CPUExecutionProvider'])
    conv_s = ort.InferenceSession(
        str(converted), so, providers=['CPUExecutionProvider'])

    rng = np.random.default_rng(0)
    feeds = build_feeds(layout, sr, input_len)

    max_diff = 0.0
    for _ in range(chunks):
        # Fresh audio each step; state carries across like streaming.
        feeds['input'] = rng.standard_normal((1, input_len)).astype(np.float32)
        stock_out = _run_stock(stock_s, layout, feeds)
        conv_out = _run_converted(conv_s, layout, feeds)
        assert len(stock_out) == len(conv_out)
        for a, b in zip(stock_out, conv_out):
            max_diff = max(max_diff, float(np.max(np.abs(a - b))))
        # Chain state: stock and converted share the same layout outputs.
        if layout.kind == 'v4':
            feeds['h'], feeds['c'] = stock_out[1], stock_out[2]
        else:
            feeds['state'] = stock_out[1]

    print(f'verify: max abs diff over {chunks} chained chunks = {max_diff}')
    # onnxsim fusion can introduce tiny float noise; keep a tight bound.
    if max_diff > 1e-5:
        raise SystemExit(
            f'converted model diverges from stock (diff={max_diff})')


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('input', type=Path, help='stock dual-rate silero_vad.onnx')
    p.add_argument('-o', '--output', type=Path, default=None,
                   help='output path (default: <stem>_<sr>k_single.onnx)')
    p.add_argument('--sr', type=int, choices=[8000, 16000], default=16000,
                   help='sample-rate branch to keep')
    p.add_argument('--window', type=int, default=None,
                   help='v4 window samples (default 512@16k / 256@8k). '
                        'Ignored for fixed-window v5/v6 except validation.')
    p.add_argument('--verify', action='store_true',
                   help='bit-exact check against the stock model in ORT')
    p.add_argument('--no-simplify', action='store_true',
                   help='skip onnxsim.simplify (default: run it)')
    args = p.parse_args(argv)

    src = args.input
    if not src.is_file():
        p.error(f'input not found: {src}')
    dst = args.output or src.with_name(
        f'{src.stem}_{args.sr // 1000}k_single.onnx')

    layout = convert(src, dst, sr=args.sr, window=args.window,
                     do_simplify=not args.no_simplify)
    window, input_len = window_and_input_len(layout, args.sr, args.window)

    if args.verify:
        verify(src, dst, layout, args.sr, window, input_len)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
