# Silero VAD in OpenVINO

The stock `silero_vad.onnx` does not load in OpenVINO. `read_model` fails with:

```
OpConversionFailure: Model wasn't fully converted.
-- Conv-16: While validating ONNX node
   '<Node(Conv): If_0_then_branch__Inline_0__/decoder/decoder/2/Conv_output_0>':
   Check 'data.get_partial_shape().rank().is_static()' failed
   The input data tensor's rank has to be known (static)
```

The ONNX frontend cannot infer static ranks for the Conv ops (the STFT) inside the If subgraphs. There is one top level If that switches between the 8 kHz and 16 kHz paths, plus 7 nested If nodes per branch left behind by TorchScript tracing, so folding the sample rate constant alone is not enough.

None of those conditions depend on the audio itself, only on the sample rate and tensor shapes, so for a fixed configuration they can all be resolved offline. `convert.py` inlines every If by evaluating its condition empirically with onnxruntime, removes the dead branch including its weights (the file drops from 2.27 MB to 1.23 MB), folds Identity nodes (recent OpenVINO serializes them as opset16::Identity, which older CPU plugins cannot execute) and fixes static shapes. The result loads and compiles in OpenVINO directly, as ONNX or as exported IR.

## Usage

```
pip install numpy onnx onnxruntime openvino

python convert.py                  # writes silero_vad_16k.onnx
python convert.py --sr 8000        # writes silero_vad_8k.onnx
python convert.py --save-ir        # also exports OpenVINO IR

python verify.py silero_vad_16k.onnx ../../tests/data/test.wav "../c++/aepyx.wav"
python verify.py silero_vad_8k.onnx --sr 8000 "../c++/aepyx_8k.wav"
```

`convert.py` defaults to the model shipped in this repo. `verify.py` compares the converted model against the stock one with chained state, on synthetic audio and on any 16 bit mono wav files you pass, and checks that the resulting speech segmentation is identical. It exits nonzero if anything is off.

## Converting older v4 / v5 exports

`convert.py` assumes the current (v5/v6) I/O: `input` / `state` / `sr` → `output` / `stateN`, with context already concatenated on `input`.

v4 uses a different contract (`input` / `h` / `c` / `sr` → `output` / `hn` / `cn`, raw window, no context). Use `convert_single_rate.py` to specialize any of v4 / v5 / v6 to a single rate with no `sr` and no `If`:

```
# download a tagged stock model first, then:
python convert_single_rate.py path/to/v4_silero_vad.onnx -o v4_16k_single.onnx --sr 16000 --window 512 --verify
python convert_single_rate.py path/to/v5_silero_vad.onnx -o v5_16k_single.onnx --sr 16000 --verify
```

`--verify` checks bit-exact agreement against the stock graph in onnxruntime over chained streaming chunks. By default the converter also runs `onnxsim.simplify`; pass `--no-simplify` to skip it.

## Set inference precision to f32

On CPUs with bf16 support (AMX or AVX512 BF16) the OpenVINO CPU plugin defaults to bf16 inference. For this model that is not a harmless accuracy tradeoff: the per chunk error compounds through the recurrent state until the speech segmentation actually changes. On real audio the default gave max abs diffs up to 3e-1 against the stock model, while f32 gives 2e-6. Always compile with:

```python
compiled = core.compile_model(model, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
```

or `ov::hint::inference_precision(ov::element::f32)` in C++. The model is tiny, f32 costs nothing.

## Inference

I/O contract of the converted model: `input` f32 [1, 576] at 16 kHz (64 context samples followed by 512 new ones; [1, 288] as 32 plus 256 at 8 kHz), `state` f32 [2, 1, 128], outputs `output` f32 [1, 1] and `stateN` f32 [2, 1, 128]. Chain `stateN` back into `state`, keep the last 64 (or 32) input samples as the next context, and reset both to zeros when a new stream starts. Same protocol as the OnnxWrapper in this repo.

```python
import numpy as np
import openvino as ov

req = ov.Core().compile_model("silero_vad_16k.onnx", "CPU",
    {"INFERENCE_PRECISION_HINT": "f32"}).create_infer_request()
state = np.zeros((2, 1, 128), np.float32)
ctx = np.zeros((1, 64), np.float32)
for chunk in stream_of_512_sample_chunks:
    x = np.concatenate([ctx, chunk[None]], axis=1)
    res = req.infer({"input": x, "state": state})
    prob, state, ctx = res["output"][0, 0], res["stateN"], x[:, -64:]
```

## Validation

Converted vs stock, chained state throughout. In onnxruntime the converted model is bit exact (max abs diff 0.0) for both sample rates, so the transformation is the mathematical identity for the chosen rate. In OpenVINO with f32: at 16 kHz, max abs diff 2.3e-06 over `tests/data/test.wav` plus `examples/c++/aepyx.wav` (229 s, 7161 chunks) with identical segmentation (29 of 29 and 65 of 65 segments). At 8 kHz, max abs diff 4.5e-06 over `examples/c++/aepyx_8k.wav` with identical segmentation (79 of 79). Reproduced on Linux and Windows across two OpenVINO versions. Source model sha256 `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3`.

## Caveats

Each converted model is single sample rate (the `sr` input is removed) with batch fixed to 1. Some of the inlined guards may be batch dependent, so batch 1 is validated exhaustively and larger batches are not. IR files are version sensitive: regenerate the IR with your target OpenVINO version, or load the cleaned ONNX directly with `read_model`.

## FAQ

You may find these answers useful.

> The OpenVINO model has to be generated separately for each major version?

No. The only version sensitive artifact is the IR (the xml plus bin pair). The cleaned ONNX that convert.py produces is a plain ONNX file and OpenVINO loads it directly with read_model, so it works across OV versions the same way the stock model works across onnxruntime versions. That's why the PR ships the conversion script instead of binaries: anyone can regenerate the cleaned ONNX from the model already in this repo in a few seconds. If you prefer to publish a converted artifact in the repo, I can add the cleaned ONNX as well. That should not require frequent updates. For IR files, I would recommend generating them locally with the target OpenVINO version, since that avoids version-compatibility edge cases.

> Is the OpenVINO model backwards of forwards compatible?

The IR is forward compatible but not backward compatible. An IR generated with an older OpenVINO keeps loading on newer runtimes (the IR format has been stable since 2022.1), but an IR generated with a newer OpenVINO can fail on older runtimes. I hit exactly that while testing this: an IR saved with 2026.2 contained opset16::Identity, which an older CPU plugin could parse but not execute. That's the reason convert.py folds Identity nodes out of the graph, and why the README says to regenerate the IR locally or just load the ONNX.

> Does it depend on CPU model?

The files don't, they are hardware agnostic and compilation for the target happens at load time. The one CPU dependent behavior is the default inference precision: on bf16 capable CPUs (AMX or AVX512 BF16, mostly server Xeons) the CPU plugin silently defaults to bf16, and for this model the error compounds through the recurrent state until the segmentation changes. With INFERENCE_PRECISION_HINT set to f32 as the README says, results match onnxruntime to about 1e-6 on every CPU I tested.

> It works only with Intel CPUs?

Not only. The CPU plugin dispatches on instruction set features rather than vendor, so it runs on AMD x86-64 as well (Intel is what gets validated and tuned), and ARM64 is officially supported too, Apple Silicon included. The Intel exclusive parts are the accelerator plugins (integrated GPU and NPU), which this example doesn't touch.
