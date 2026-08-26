# Silero VAD 16 kHz 单速率 ONNX 模型：输入输出与使用说明

本文描述仓库中三个已经特化到 **16 kHz、batch=1、无 `sr` 输入** 的 ONNX 模型：

| 文件 | 来源权重 | 转换脚本 | 体积 |
| --- | --- | --- | --- |
| `examples/openvino/models/v4_16k_single.onnx` | 官方 v4 `silero_vad.onnx` | `convert_single_rate.py`（含 onnxsim） | 621 KiB |
| `examples/openvino/models/v5_16k_single.onnx` | 官方 v5 `silero_vad.onnx` | `convert_single_rate.py`（含 onnxsim） | 1.19 MiB |
| `src/silero_vad/data/silero_vad_openvino_16k.onnx` | 当前仓库 `silero_vad.onnx`（v5/v6 I/O，现行权重） | `convert.py`（无 onnxsim） | 1.23 MiB |

三者都可以用 ONNX Runtime 或 OpenVINO 直接加载。它们都已经去掉采样率分支（`If`）和 `sr` 输入，并把形状冻成静态，因此可以避开官方双速率图在 OpenVINO 上无法推断 Conv 静态 rank 的问题。

**不要把 `v5_16k_single.onnx` 和 `silero_vad_openvino_16k.onnx` 当成同一套权重。** 二者 I/O 契约相同，但权重不同，数值结果不可互换。各自与对应的源模型在 ONNX Runtime 上流式串联状态时是 **bit-exact**（max abs diff = 0）。

---

## 1. 总览对比

| 项目 | `v4_16k_single.onnx` | `v5_16k_single.onnx` | `silero_vad_openvino_16k.onnx` |
| --- | --- | --- | --- |
| 架构代际 | Silero VAD v4 | Silero VAD v5 | 当前仓库模型（v5/v6 I/O） |
| 采样率 | 16 kHz 专用 | 16 kHz 专用 | 16 kHz 专用 |
| 每步新音频 | 512 samples（32 ms） | 512 samples（32 ms） | 512 samples（32 ms） |
| 是否拼接 context | **否**，直接喂窗口 | **是**，64 + 512 | **是**，64 + 512 |
| `input` 形状 | `float32 [1, 512]` | `float32 [1, 576]` | `float32 [1, 576]` |
| RNN 状态 | 分离的 `h` / `c` | 堆叠的 `state` | 堆叠的 `state` |
| 状态形状 | `h,c: float32 [2, 1, 64]` | `state: float32 [2, 1, 128]` | `state: float32 [2, 1, 128]` |
| LSTM | 2 层，hidden=64 | 1 层，hidden=128 | 1 层，hidden=128 |
| 输出概率 | `output: float32 [1, 1]` | 同左 | 同左 |
| 下一状态 | `hn`, `cn` | `stateN` | `stateN` |
| 图节点 / initializer | 73 / 54 | 35 / 27 | 167 / 0（权重在 Constant 节点里） |
| ONNX opset | 16 | 16 | 16 |
| 控制流节点 | 无 `If`/`Loop`/`Scan` | 无 | 无 |
| `sr` 输入 | 已删除 | 已删除 | 已删除 |
| batch | 固定 1 | 固定 1 | 固定 1 |

输入输出名称必须按上表精确匹配；ONNX Runtime / OpenVINO 都按名字喂张量。

---

## 2. 音频预处理（三个模型共用）

模型内部已经包含 STFT / 编码器，**不要**再自己做梅尔谱或额外标准化。调用方只需要提供波形。

| 项目 | 要求 |
| --- | --- |
| 声道 | 单声道。多声道需先混音或取一轨 |
| 采样率 | **必须 16 000 Hz**。这些图已经砍掉 8 kHz 分支，错采样率会得到无意义结果 |
| dtype | `float32` |
| 数值范围 | PCM 线性幅度，官方封装使用 `int16 / 32768.0`，即大约 `[-1, 1]` |
| 布局 | `[batch=1, time]`，时间维在最后 |
| 步进 | 非重叠窗口，每 32 ms 推进一步（512 samples） |
| 尾块 | 不足 512 时在右侧零填充 |

16 kHz 下的时间换算：

- 512 samples = 32 ms（模型真正“看见”的新音频）
- 64 samples = 4 ms（仅 v5 / 现行模型的 context）
- 576 samples = 36 ms（v5 / 现行模型一次推理的 `input` 长度）

---

## 3. `v4_16k_single.onnx`

### 3.1 输入

| 名称 | dtype | 形状 | 含义 |
| --- | --- | --- | --- |
| `input` | `float32` | `[1, 512]` | 当前 32 ms 单声道波形。**不要**拼接上一窗尾部 |
| `h` | `float32` | `[2, 1, 64]` | 2 层 LSTM 的 hidden state。dim0=层数，dim1=batch，dim2=hidden |
| `c` | `float32` | `[2, 1, 64]` | 对应的 cell state |

新流开始时：

```python
h = np.zeros((2, 1, 64), np.float32)
c = np.zeros((2, 1, 64), np.float32)
```

官方 v4 在 16 kHz 上还支持 1024 / 1536 窗口；**本文件已把长度冻成 512**，喂其他长度会失败。

### 3.2 输出

| 名称 | dtype | 形状 | 含义 |
| --- | --- | --- | --- |
| `output` | `float32` | `[1, 1]` | 当前窗语音概率，经过 Sigmoid，范围约 `[0, 1]` |
| `hn` | `float32` | `[2, 1, 64]` | 下一窗的 `h` |
| `cn` | `float32` | `[2, 1, 64]` | 下一窗的 `c` |

### 3.3 流式协议

1. 新音频流：`h`、`c` 置零。
2. 按 512 samples 切块（尾块右零填）。
3. 推理：`input, h, c` → `output, hn, cn`。
4. 把 `hn`/`cn` 原样写回 `h`/`c`。
5. 换文件 / 换说话人 / 中断后重开流时必须重新置零，不能跨流复用状态。

v4 **没有** 64-sample context。`input` 就是当前窗本身。

### 3.4 示例（ONNX Runtime）

```python
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession("examples/openvino/models/v4_16k_single.onnx",
                            providers=["CPUExecutionProvider"])
h = np.zeros((2, 1, 64), np.float32)
c = np.zeros((2, 1, 64), np.float32)

# wav: 1-D float32, 16 kHz, ~[-1, 1]
probs = []
for i in range(0, len(wav), 512):
    chunk = wav[i:i + 512]
    if len(chunk) < 512:
        chunk = np.pad(chunk, (0, 512 - len(chunk)))
    out, h, c = sess.run(
        ["output", "hn", "cn"],
        {"input": chunk[None].astype(np.float32), "h": h, "c": c},
    )
    probs.append(float(out[0, 0]))
```

---

## 4. `v5_16k_single.onnx` 与 `silero_vad_openvino_16k.onnx`

这两个文件的 **I/O 名字、dtype、形状、context 规则完全相同**，调用代码可以共用；只是权重和图优化程度不同。

- `v5_16k_single.onnx`：从 tagged v5 导出图经 `convert_single_rate.py` + onnxsim 得到，节点少、权重量化为 graph initializer。
- `silero_vad_openvino_16k.onnx`：从当前仓库 `src/silero_vad/data/silero_vad.onnx` 经 `convert.py` 得到，权重散落在 `Constant` 节点中。这是与 `OnnxWrapper` / `get_speech_timestamps` 现行行为对齐的那一份。

新项目如果要复现本仓库 Python API 的结果，应使用 `silero_vad_openvino_16k.onnx`。若需要与历史 v5 导出对齐，才用 `v5_16k_single.onnx`。

### 4.1 输入

| 名称 | dtype | 形状 | 含义 |
| --- | --- | --- | --- |
| `input` | `float32` | `[1, 576]` | **先 64 点 context，再 512 点新音频**，已经拼好 |
| `state` | `float32` | `[2, 1, 128]` | LSTM 状态。`state[0]` 为 hidden `h`，`state[1]` 为 cell `c`（`torch.stack([h, c], dim=0)`） |

新流开始时：

```python
state = np.zeros((2, 1, 128), np.float32)
ctx = np.zeros((1, 64), np.float32)   # 第一窗的左侧 context
```

`input` 的时间轴布局：

```
index:   0 ........... 63 | 64 .......... 575
         <--- context ---> <--- 当前 512 点 --->
         上一窗最后 64 点     本窗新音频
```

第一窗的 context 全零，等价于在波形左侧 pad 4 ms 静音。这与仓库 `OnnxWrapper`、`examples/cpp/silero-vad-onnx.cpp` 一致。

### 4.2 输出

| 名称 | dtype | 形状 | 含义 |
| --- | --- | --- | --- |
| `output` | `float32` | `[1, 1]` | 当前 32 ms 窗的语音概率，Sigmoid 后约 `[0, 1]` |
| `stateN` | `float32` | `[2, 1, 128]` | 下一窗的 `state`，必须原样回灌 |

时间戳应对齐 **当前 512 点新音频**，不要把左侧 64 点 context 算进本窗时长。第 `i` 窗（从 0 计）对应采样点 `[i*512, (i+1)*512)`，时间 `[i*0.032, (i+1)*0.032)` 秒。

### 4.3 流式协议

1. 新流：`state` 全零，`ctx` 全零。
2. 取下一块 512 samples（尾块右零填）。
3. `x = concat(ctx, chunk)` → 形状 `[1, 576]`。
4. 推理：`input=x, state=state` → `output, stateN`。
5. `state = stateN`。
6. `ctx = x[:, -64:]`（即本窗最后 64 点，供下一窗使用）。
7. 换流时同时清零 `state` 和 `ctx`。

漏掉 context 拼接、或者把 `stateN` 清零后再喂下一窗，概率会漂，分段会错。

### 4.4 示例（ONNX Runtime）

```python
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession(
    "src/silero_vad/data/silero_vad_openvino_16k.onnx",  # 或 v5_16k_single.onnx
    providers=["CPUExecutionProvider"],
)
state = np.zeros((2, 1, 128), np.float32)
ctx = np.zeros((1, 64), np.float32)

probs = []
for i in range(0, len(wav), 512):
    chunk = wav[i:i + 512]
    if len(chunk) < 512:
        chunk = np.pad(chunk, (0, 512 - len(chunk)))
    x = np.concatenate([ctx, chunk[None]], axis=1)  # [1, 576]
    out, state = sess.run(
        ["output", "stateN"],
        {"input": x.astype(np.float32), "state": state},
    )
    ctx = x[:, -64:]
    probs.append(float(out[0, 0]))
```

### 4.5 示例（OpenVINO）

CPU 上若支持 bf16（AMX / AVX512 BF16），OpenVINO CPU plugin 默认会走 bf16。对本模型这不是无害的精度交换：逐步误差会经 LSTM 状态累积，最终改变语音分段。编译时必须强制 f32：

```python
import numpy as np
import openvino as ov

compiled = ov.Core().compile_model(
    "src/silero_vad/data/silero_vad_openvino_16k.onnx",
    "CPU",
    {"INFERENCE_PRECISION_HINT": "f32"},
)
req = compiled.create_infer_request()

state = np.zeros((2, 1, 128), np.float32)
ctx = np.zeros((1, 64), np.float32)
for i in range(0, len(wav), 512):
    chunk = wav[i:i + 512]
    if len(chunk) < 512:
        chunk = np.pad(chunk, (0, 512 - len(chunk)))
    x = np.concatenate([ctx, chunk[None]], axis=1)
    res = req.infer({"input": x.astype(np.float32), "state": state})
    prob = float(res["output"][0, 0])
    state = res["stateN"]
    ctx = x[:, -64:]
```

C++ 对应 `ov::hint::inference_precision(ov::element::f32)`。

---

## 5. 从概率到语音时间戳

模型每窗只给出一个标量概率。官方后处理（`get_speech_timestamps` / `VADIterator`）的默认超参可直接套用：

| 参数 | 默认 | 作用 |
| --- | --- | --- |
| `threshold` | `0.5` | 进入语音：`prob >= 0.5` |
| `neg_threshold` | `threshold - 0.15` → `0.35` | 离开语音：已在语音态且 `prob < 0.35` |
| `min_speech_duration_ms` | 250 | 短于此时长的片段丢弃 |
| `min_silence_duration_ms` | 100 | 语音结束后再等这段静音才切段 |
| `speech_pad_ms` | 30 | 段首/段尾各外扩 |

实时场景用 `VADIterator` 同类状态机即可：概率过 `threshold` 报 start，连续低于 `neg_threshold` 且静音够长后报 end。

---

## 6. 常见错误

1. **v5 / 现行模型只喂 512 点**  
   图期望 `[1, 576]`。必须自行 concat 64 点 context。v4 才是 `[1, 512]`。

2. **把 v4 的 `h`/`c` 和 v5 的 `state` 混用**  
   形状和语义都不同：`[2,1,64]`×2 vs `[2,1,128]`。

3. **跨文件复用 RNN 状态**  
   每个独立音频流都要重置状态（以及 v5 的 context）。

4. **采样率不是 16 kHz**  
   这些文件没有 `sr` 输入，8 kHz 音频不会自动走另一条分支。需要先重采样。

5. **OpenVINO 未强制 f32**  
   在 bf16 CPU 上分段结果可能与 ORT 不一致。始终设置 `INFERENCE_PRECISION_HINT=f32`。

6. **batch > 1**  
   转换时把 batch 冻成 1，部分被内联的 TorchScript 守卫与 batch 有关。不要喂更大的 batch。

7. **把 `v5_16k_single` 当现行官方模型**  
   与 `silero_vad_openvino_16k.onnx` / 仓库 `silero_vad.onnx` 权重不同，概率序列会对不齐。

---

## 7. 与官方双速率图的关系

源模型（`v4_silero_vad.onnx`、`v5_silero_vad.onnx`、`src/silero_vad/data/silero_vad.onnx`）额外带一个标量 `sr: int64 []`，并在图内用 `If` 切换 8 kHz / 16 kHz。Python 封装 `OnnxWrapper` 对现行双速率图的调用是：

```python
ort_inputs = {
    "input": x.numpy(),          # 已 concat context，16 kHz 时 [B, 576]
    "state": self._state.numpy(), # [2, B, 128]
    "sr": np.array(sr, dtype="int64"),
}
```

单速率转换做了三件事：内联全部 `If`、丢掉未用采样率分支、删除 `sr`。因此：

- 喂给转换模型的 `input` / 状态张量，与喂给对应源模型 16 kHz 路径的张量 **完全一样**；
- 只是不再传 `sr`；
- 在 ONNX Runtime 上，转换图与源图对同一串流式窗是逐位一致的。

重新生成：

```bash
# v4 / v5 tagged 导出
python convert_single_rate.py path/to/v4_silero_vad.onnx -o v4_16k_single.onnx --sr 16000 --window 512 --verify
python convert_single_rate.py path/to/v5_silero_vad.onnx -o v5_16k_single.onnx --sr 16000 --verify

# 当前仓库模型（默认读 src/silero_vad/data/silero_vad.onnx）
python convert.py -o ../../src/silero_vad/data/silero_vad_openvino_16k.onnx
```
