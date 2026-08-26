# Silero VAD v4 / v5 / v6 与 TEN-VAD 评估协议

本文说明 `eval_vad_datasets.py` 在 AISHELL-5 与 VoxConverse 上对比 Silero v4、v5、v6 以及 **TEN-VAD** 时实际采用的模型、**评了哪些数据**、语音/非语音划分，以及每一步为什么这样设计。实现以脚本为准；数值结果见 [`eval_vad_results.md`](eval_vad_results.md)（机器可读原文 `eval_vad_results.json`）。

Silero 评估对象是**官方双速率 stock ONNX**，不是 `convert_single_rate.py` / `convert.py` 产出的 16 kHz 单速率图。TEN-VAD 用本地 `D:\ten-vad` 的 Windows x64 DLL，hop **256**（16 ms）。数据集范围见第 2 节：只用 AISHELL-5 的 Dev/Eval1/Eval2（近场 `DA*` + 远场 `DX01–04`）和 VoxConverse 0.3 的全部 dev+test，16 kHz。

真值规则（并集、50% 重叠）四个模型相同。帧格子随 hop 变：Silero 31.25 ms，TEN 16 ms。AUC / Acc / F1 因此不是同一套时间轴上的数，但仍是各自原生窗上的帧级微平均，适合看「这个模型按它自己的步长判得怎么样」。

---

## 1. 评估用模型路径

脚本里的映射（Silero 相对仓库根目录 `silero-vad/`，TEN 为绝对路径）：

| 名称 | 路径 | 角色 | hop | SHA-256 |
| --- | --- | --- | ---: | --- |
| v4 | `examples/openvino/models/v4_silero_vad.onnx` | 官方 v4 双速率导出（`input/sr/h/c`） | 512 | `a35ebf52fd3ce5f1469b2a36158dba761bc47b973ea3382b3186ca15b1f5af28` |
| v5 | `examples/openvino/models/v5_silero_vad.onnx` | 官方 v5 双速率导出（`input/state/sr`） | 512 | `2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f` |
| v6 | `src/silero_vad/data/silero_vad.onnx` | 当前仓库随包装载的权重（v5/v6 I/O，v6.x） | 512 | `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3` |
| ten | `D:\ten-vad\lib\Windows\x64\ten_vad.dll` | TEN-VAD Windows x64 预编译库 | 256 | `38937f5604fa93a7941db7b9326992b792fa3731ebf9353973b3234457c6064b` |

**不要**把下面这些文件当成本次评估模型：

| 文件 | 为什么不是 |
| --- | --- |
| `examples/openvino/models/v4_16k_single.onnx` | v4 的 16 kHz 特化图，评估未使用 |
| `examples/openvino/models/v5_16k_single.onnx` | v5 的 16 kHz 特化图，评估未使用 |
| `src/silero_vad/data/silero_vad_openvino_16k.onnx` | 现行权重的 OpenVINO 友好图，评估未使用 |
| `src/silero_vad/data/silero_vad.jit` | TorchScript，未参与对比 |
| `src/silero_vad/data/silero_vad_16k_op15.onnx` | 仅 16 kHz、opset 15 的精简导出 |

v5 与 v6 的 ONNX 体积同为 2 327 524 字节，但哈希不同，权重不是同一份。

Silero 由 ONNX Runtime `CPUExecutionProvider` 加载；TEN-VAD 经 `D:\ten-vad\include\ten_vad.py` 加载 `ten_vad.dll`。每条音频按各自流式契约逐步推进内部状态（见第 5.6 节）。阈值类指标默认用 `0.5`。只评 TEN 时：`python examples/eval_vad_datasets.py --models ten`，结果会合并进已有 JSON，不覆盖 v4/v5/v6。

---

## 2. 评估数据集范围

本次评测只用本地这两份数据，没有抽样、没有另做子集。凡进入指标的 wav，都必须能配对到标注，且采样率为 16 kHz 单声道 PCM16。

| 条件名（脚本 `dataset`） | 数据来源 | 根目录 | 用到的官方划分 | 文件数 | 时长 | 真值语音占比 |
| --- | --- | --- | --- | --- | --- | --- |
| `aishell5_near` | AISHELL-5 近场耳机麦 | `D:\AISHELL-5-Data` | `Dev` + `Eval1` + `Eval2` | 108 | 22.77 h | 45.1% |
| `aishell5_far` | AISHELL-5 远场车门麦 | `D:\AISHELL-5-Data` | `Dev` + `Eval1` + `Eval2` | 216 | 45.53 h | 79.2% |
| `voxconverse` | VoxConverse 0.3 | `D:\voxconverse` | `dev` + `test` | 448 | 63.83 h | 90.7% |
| **合计** | | | | **772** | **132.13 h** | |

「文件数」按 wav 计：AISHELL-5 远场同一会话的四个车门麦是四条独立样本。下面按数据集写清**收录规则、split 明细、磁盘上有但没用的部分**。

### 2.1 AISHELL-5

磁盘布局（评估只进前三个子目录）：

```text
D:\AISHELL-5-Data\
  Dev\          ← 用
  Eval1\        ← 用
  Eval2\        ← 用
  train\        ← 不用
  noise\        ← 不用
  *.tar.gz、论文 PDF  ← 不用
```

每个 split 下是会话目录 `001`–`018`（共 18 场）。一场里常见文件：

| 文件名 | 含义 | 是否评估 |
| --- | --- | --- |
| `DA01.wav` / `DA02.wav` / `DA03.wav` / `DA04.wav` 及同名 `.TextGrid` | 近场耳机麦，一人一轨 | **用**（有 TextGrid 才进 `aishell5_near`） |
| `DX01C01.wav`–`DX04C01.wav` 及同名 `.TextGrid` | 四个车门上方的远场麦 | **用**（有 TextGrid 才进 `aishell5_far`） |
| `DX05C01.wav` / `DX06C01.wav` | 额外远场通道 | **不用**（磁盘上没有配套 TextGrid） |
| 有 wav、无 TextGrid 的其它通道 | — | **不用** |

脚本收录条件（三条同时满足）：

1. 路径在 `Dev` / `Eval1` / `Eval2` 下（默认 `--aishell-splits Dev Eval1 Eval2`）。
2. 存在**同一目录、同一 stem** 的 `.TextGrid`。
3. stem 属于近场（以 `DA` 开头）或远场集合 `{DX01C01, DX02C01, DX03C01, DX04C01}`。

split 明细（与 `eval_vad_results.json` 一致）。近场、远场分开列，避免空单元格把列对乱。

近场 `aishell5_near`（每场 2 路耳机麦，会话 `001`–`018`）：

| split   | 目录     | 文件数 | 时长    | 语音占比 |
| ------- | -------- | -----: | ------: | -------: |
| `dev`   | `Dev/`   |     36 |  7.86 h |    42.4% |
| `eval1` | `Eval1/` |     36 |  7.13 h |    47.0% |
| `eval2` | `Eval2/` |     36 |  7.78 h |    46.1% |
| 小计    | -        |    108 | 22.77 h |    45.1% |

远场 `aishell5_far`（同一 18 场 × 4 路车门麦 `DX01C01`–`DX04C01`）：

| split   | 目录     | 文件数 | 时长    | 语音占比 |
| ------- | -------- | -----: | ------: | -------: |
| `dev`   | `Dev/`   |     72 | 15.71 h |    75.8% |
| `eval1` | `Eval1/` |     72 | 14.26 h |    81.0% |
| `eval2` | `Eval2/` |     72 | 15.56 h |    81.0% |
| 小计    | -        |    216 | 45.53 h |    79.2% |

近场每个 split 是 36 条 = 18 场 × **恰好 2 条** `DA*`：Dev / Eval1 / Eval2 里每一场都只有两路有标签的近场耳机麦（不是 18×4）。每条 `DA*.wav` 的 TextGrid 只有 **一个** 说话人 tier，所以近场评估是「单人近讲」；一场会话里纳入的是两个人，不是四个人。`DA01`–`DA04` 对应座位，哪两路出现因场次而异（例如有的是 `DA03+DA04`，有的是 `DA01+DA03`）。论文里「一场 2–4 人」描述的是整份 AISHELL-5（含 train）；**本评测实际用到的 Dev/Eval 近场全部是每场 2 人**。远场三个 split 都是 18×4=72，因为 `DX01C01`–`DX04C01` 均有 TextGrid；远场 TextGrid 也是 2 个说话人 tier，与这两路近场对应。

**明确不在范围内：**

| 未用部分 | 原因 |
| --- | --- |
| `train/`（约 94 h，近场+远场） | 官方训练划分；拿来评测会混进见过的数据 |
| `noise/`（无说话人的车内噪声，约 40 h） | 没有语音活动真值，本协议不做「整段是否误检」的纯噪声集评估 |
| `DX05C01` / `DX06C01` | 无 TextGrid |
| 任意无配套 TextGrid 的 wav | 无法监督 |
| AISHELL-4 | 本地没有；wiki 上的 AISHELL-4 数字不能当本表基线 |

语音/非语音如何从 TextGrid 得到，见第 4.2 节。

### 2.2 VoxConverse

版本按仓库 README 为 **0.3**（test 的 RTTM 相对 0.2 有过修正）。磁盘布局：

```text
D:\voxconverse\
  dev\                    ← 216 个 *.rttm，全部用
  test\                   ← 232 个 *.rttm，全部用
  voxconverse_dev_wav\    ← 与 dev rttm 按 stem 配对的 wav
  voxconverse_test_wav\   ← 与 test rttm 按 stem 配对的 wav
  *.zip、README.md        ← 不用
```

脚本收录条件（默认 `--vox-splits dev test`）：

1. `dev/*.rttm` 或 `test/*.rttm` 中的每一条都尝试配对。
2. 在对应 wav 根目录下递归查找 **同 stem** 且文件名不以 `._` 开头的 `.wav`。
3. 找不到 wav 的 rttm 丢弃（本机上没有这种情况）。

| split | 标注 | 音频目录 | 纳入评估 | 时长 | 语音占比 |
| --- | --- | --- | ---: | ---: | ---: |
| `dev` | `dev/*.rttm` 216 条 | `voxconverse_dev_wav/**/*.wav` | 216 | 20.30 h | 93.2% |
| `test` | `test/*.rttm` 232 条 | `voxconverse_test_wav/**/*.wav` | 232 | 43.54 h | 89.5% |
| **合计** | | | **448** | **63.83 h** | 90.7% |

test 目录里实际有约 464 个 wav：其中 232 个是 `._stem.wav` 形式的 macOS 资源叉，脚本按文件名前缀 `._` 丢掉，因此 **test 用的是完整官方 232 条**，时长 43.536 h，与 Silero wiki 的 VoxConverse test 43.5 h 对齐。dev 不在 wiki 主表里，本评测仍完整跑完，并单独出数，同时把它当作该数据集上搜 F1 阈值的验证集。

配对例子：`test/aepyx.rttm` ↔ `voxconverse_test_wav/voxconverse_test_wav/aepyx.wav`。

**明确不在范围内：**

| 未用部分 | 原因 |
| --- | --- |
| `._*.wav` | 不是有效 PCM |
| `*.zip` | 压缩包，不是评估输入 |
| 没有对应 rttm 的 wav | 无真值 |
| 说话人 ID、重叠说话人计数 | RTTM 里有 `spk00`/`spk01` 以及同一时刻多人说话。评测不做说话人日志（不管是谁），也不做重叠检测（不管几个人）。两人同时说仍并成一段「有语音」，见第 4.3 节 |

### 2.3 范围一句话

评测覆盖：AISHELL-5 的 **Dev + Eval1 + Eval2** 上全部有标签的近场 `DA*` 与远场 `DX01–04`，以及 VoxConverse 0.3 的 **全部 dev + 全部 test**。不覆盖 AISHELL-5 的 `train`/`noise`、无 TextGrid 的通道、VoxConverse 的资源叉文件，也不覆盖 8 kHz。

---

## 3. 评估策略（结论先行）

对齐 [Silero Quality Metrics](https://github.com/snakers4/silero-vad/wiki/Quality-Metrics) 的帧级协议，而不是说话人日志或 `get_speech_timestamps` 的分段后处理：

1. 音频统一为 **16 kHz 单声道**。磁盘上的 wav 是 **PCM16**。Silero 读入后除以 `32768.0` 变成 **float32**（约 `[-1, 1]`）；TEN-VAD 官方契约吃 **int16**，脚本用 `soundfile` 直接读 `int16`，不再先转浮点。本批数据已经是 16 kHz PCM16，**不做重采样、不做峰值归一化**。
2. 把整段波形切成互不重叠的 hop 窗。Silero hop **512**（**31.25 ms**）；TEN-VAD hop **256**（**16 ms**，官方推荐配置之一）。不足一窗的尾块右侧补零（TEN 官方 demo 会丢掉不足 hop 的尾部；本评测与 Silero 一样补零并打标，好让最后一窗也进入指标）。标签只按真实音频时长计算重叠。
3. 模型对每一窗输出一个语音概率 `p ∈ [0, 1]`。状态在同一条音频内串联，换文件时清零（TEN 是每个文件新建 handle）。
4. 真值落到**该模型自己的 hop 格子**上：一窗与**某一段连续**真值语音重叠 **≥ 50%** 则为语音帧（`1`），否则为非语音帧（`0`）。详见第 4.4 节。
5. **主指标是 ROC-AUC**（不依赖阈值）。同时报告默认阈值 0.5 下的 Accuracy、Precision、Recall、F1、FAR、MR，以及推理 RTF。
6. 帧在全集上 **微平均**（所有文件的帧拼在一起再算），不先按文件平均再平均。
7. 不用 NIST collar、不用 `min_speech_duration_ms` / `min_silence_duration_ms` / `speech_pad_ms`。那些是部署分段启发式，会掩盖模型本身的帧级判别力。

官方 wiki 写的是 31.25 ms 段上的 ROC-AUC 与 Accuracy。Silero 的 512 / 16000 = 0.03125 s，与 wiki 一致。TEN 按官方 hop 256 / 16000 = 0.016 s。部分文档把 512 点约成 32 ms，Silero 评估实现按 31.25 ms 计时。

---

## 4. 数据集怎么切成「语音 / 非语音」

两个数据集都不是现成的帧级 VAD 标签。VAD 只问「这一刻有没有人在说话」，不问「是谁在说」。因此一律把多说话人标注 **并成一条时间轴上的语音活动**，重叠说话只算一次语音。

**先分清两件事：** 模型只吃波形；0/1 标签是事后用来打分的，不进网络。

**模型输入（喂进网络）**

- 来源：wav
- 内容：16 kHz 单声道波形
- 步进：每步新音频 512 点（31.25 ms）；v5/v6 在前面再拼上一段 64 点 context

**评估真值（不喂进网络）**

- 来源：AISHELL-5 的 TextGrid，或 VoxConverse 的 RTTM
- 内容：与上面每一窗 512 点对齐的 `1` / `0`（语音 / 非语音）
- 用途：模型先对每窗算出概率 `p`，再用这些 0/1 去算 AUC、Accuracy 等；规则见下面 4.1–4.4 节

进入本步的文件集合就是第 2 节划定的范围。

### 4.1 公共规则

| 步骤 | 规则 | 原因 |
| --- | --- | --- |
| 读区间 | 得到若干 `[start, end)` 秒 | 两个数据集的原始标注都是时间区间 |
| 过滤 | `end > start` 才保留 | 丢掉空区间 |
| 合并 | 按起点排序，相邻或重叠区间并成一段 | 重叠说话对 VAD 仍是「有语音」；不合并会在帧上重复计数重叠 |
| 切窗打标 | 见第 4.4 节：每 512 点一窗，重叠 ≥ 50% 标成语音 | 模型输出是帧，真值必须在同一分辨率上 |

### 4.2 AISHELL-5：TextGrid

文件范围见第 2.1 节。每个会话目录里，近场 TextGrid 通常只有 **一个 IntervalTier**（戴该耳机的说话人）；远场 TextGrid 常有 **两个或多个 IntervalTier**（车内所有人）。脚本对 **所有 tier 一视同仁**：

```
区间 text 去掉首尾空白后非空  →  语音
区间 text 为空（Praat 的 ""）  →  非语音
```

非空文本包括正常转写、`*情景三。` 这类场景提示音、语气词等。原因：TextGrid 里「有字」表示标注者认为这段是人声活动；空串才是静音/未转写的非语音。脚本不把 `*`、噪音标记再拆成第三类，以免引入主观规则。

多 tier 的区间全部收集后再 `merge_intervals`。因此远场一条 `DX01C01.wav` 的真值是 **整车「有人在说」**，不是「这个麦正对的那个人在说」。这是有意的：

- VAD 任务定义是语音活动，不是波束指向或说话人分配。
- 远场 TextGrid 是会话级转写拷到各通道上的，并不是「该麦实际听得见的人」。
- 后果：远端座位上几乎听不清的人仍会被标成语音。远场 AUC 会明显低于近场。这反映车内单通道 VAD 的真实难度，而不是标注 bug。若只想衡量「本通道可听见的语音」，需要按通道能量再过滤真值；**本脚本没有做这一步**。

split 名称在脚本里记成小写 `dev` / `eval1` / `eval2`。三个 split 都单独出数；`dev` 另外用于搜 F1 最优阈值（见第 5.7 节），主指标仍用阈值 0.5。

### 4.3 VoxConverse：RTTM

文件范围见第 2.2 节。RTTM 行格式：

```
SPEAKER <utt> 1 <start> <duration> <NA> <NA> <spk> <NA> <NA>
```

只读 `SPEAKER` 行，取 `start` 与 `start+duration`。所有说话人的区间并集 = 语音；空隙 = 非语音。原因：VoxConverse 是说话人日志数据集，官方 Silero wiki 把它当 VAD 测试集时同样是「有人说话 vs 没有」，不是 DER。

### 4.4 每一 hop 窗如何标成语音或静音

实现是 `intervals_to_frames()`。一条 wav 有 **N** 个采样点（16 kHz），hop 记为 **H**（Silero H = 512，TEN H = 256），切成

```text
F = ceil(N / H)
```

个互不重叠的窗。第 **i** 窗（i = 0, 1, …, F−1）对应：

| 项目 | 定义 |
| --- | --- |
| 采样点下标 | `[H·i, H·i+H)`，右开 |
| 墙上时钟 | `[t0, t1) = [i × H/16000, (i+1) × H/16000)` 秒 |
| 窗长 | 满窗 Silero **31.25 ms**，TEN **16 ms**。最后一窗若 N 不是 H 的倍数，真实音频只覆盖到 N/16000 秒，右侧用 0 填满 H 点再送给模型；**打标只用真实覆盖的时长**，不把补零算进分母 |

合并后的真值语音是若干**互不重叠**的区间 `S = {[sk, ek)}`（重叠说话已经在上一步并成一段；中间隔了静音的两段说话仍是两段）。第 i 窗与其中第 k 段的重叠为

```text
ov[i,k] = max(0, min(t1, ek) − max(t0, sk))
```

最后一窗的 t1 改成真实音频终点 N/16000，有效窗长 `di = t1 − t0`（满窗则 di = H/16000）。补零那一段不计入 di。

**判定（对某一段连续语音做多数投票，实现见 `intervals_to_frames`）：**

```text
若存在某一段 k，使得 ov_{i,k} >= 0.5 * d_i
        →  labels[i] = 1   （语音）
否则    →  labels[i] = 0   （非语音 / 静音）
```

一窗标成语音，当且仅当 **至少有一段连续的真值语音盖住了该窗一半及以上的时间**。刚好 50%（≥）算语音。

中间夹了静音的两段说话**不会把重叠加总**：例如一窗 31.25 ms 里前 10 ms 说话、中间 11 ms 静音、后 10 ms 说话，两段各自都不到 15.625 ms，标签仍是 `0`。这与代码一致（按段分别比 50%，不是先把该窗里所有语音毫秒加起来再比）。

数值例子（以 Silero 满窗 31.25 ms，di = 31.25 ms 为例；TEN 满窗是 16 ms，50% 阈值按 8 ms 同比）：

| 该窗与某一段连续真值语音的重叠 | 标签 | 含义 |
| ---: | --- | --- |
| 0 ms | `0` 静音 | 整窗都在无人说的区间里 |
| 10 ms（32%） | `0` 静音 | 只有边角碰到说话，主导内容仍是静音 |
| 15.625 ms（50%） | `1` 语音 | 恰好一半，判语音 |
| 31.25 ms（100%） | `1` 语音 | 整窗都在说话区间里 |

再举边界：真值语音在 1.000–2.000 s。第 32 窗覆盖 1.000–1.03125 s，整窗在语音里 → `1`。第 63 窗覆盖 1.96875–2.000 s，整窗在语音里 → `1`。第 64 窗覆盖 2.000–2.03125 s，重叠 0 → `0`。若某窗 1.990–2.02125 s，重叠 10 ms < 15.625 ms → `0`。

**标签不是这样来的（本脚本都没做）：**

| 未采用的打标法 | 本协议为何不用 |
| --- | --- |
| 看模型概率再标 | 那是预测，不是真值 |
| 看这一窗 RMS / 能量门限 | 能量高也可能是音乐、引擎；真值来自人工区间 |
| 窗中心点落在语音里就算语音 | 满窗时与 50% 重叠几乎一样，缩短的尾窗上不如按有效时长比例明确 |
| 与真值有任何重叠就算语音 | 边界帧会被大量打成语音，虚高召回 |
| NIST collar（边界 ±200/250 ms 忽略） | 官方 wiki 不用，加上就无法对照 VoxConverse 数字 |
| `get_speech_timestamps` 的最短语音/静音/pad | 那是部署分段，不是帧标签 |

默认阈值 **0.5** 只用于把模型输出的概率变成「预测语音/预测静音」，去和上面的 `labels[i]` 比。它 **不参与** 真值怎么打；真值只由标注区间和 50% 规则决定。

---

## 5. 评估流程（按决策顺序）

下面按当时落地脚本的思考顺序写。每一步都写「做了什么」和「为什么不选另一种做法」。

### 5.1 先定任务：帧级 VAD，不是分段、不是日志

Silero 模型的直接输出是每窗一个概率，官方质量表也是 31.25 ms 上的 ROC-AUC / Accuracy。若先跑 `get_speech_timestamps`（滞后阈值、最短语音、最短静音、两端 pad），再拿分段去对 RTTM，测到的是「模型 + 一套超参」的部署效果，三个版本无法在同一后处理上公平对比，也无法和 wiki 对表。

因此脚本只保留 **原始帧概率 vs 帧标签**。

### 5.2 用 stock ONNX，不用单速率转换图、不用 JIT

v4 与 v5/v6 的 I/O 不同（v4：`h`/`c`、无 context；v5/v6：`state`、输入为 64+512 context）。转换图已经验证与源模型 bit-exact，但评估仍加载第 1 节的三份 stock 文件，并在脚本里分别实现两种前向。原因：读者看到的数字对应「官方发布的图」，不依赖本仓库的 OpenVINO 转换是否被接受。

### 5.3 数据只用测试向划分，并把 AISHELL-5 拆成近场 / 远场

早期只用远场 `DX*` 试跑时，单条会话上语音/非语音的平均概率几乎重合（约 0.28 vs 0.26），AUC 接近随机。同一会话的近场 `DA*` 则分明（约 0.80 vs 0.07）。说明模型没坏，而是 **远场真值含听不清的远端说话人**。

若只报远场，会把「车内单通道听不清」写成「v4/v5/v6 都不能用」。因此拆成两个条件：

- 近场：衡量模型在中文、较高 SNR 上的帧级质量。
- 远场：衡量无前端（无 AEC/IVA）时的车内单通道 VAD。AISHELL-5 论文基线本身也是先分离再 VAD。

四个远场通道都保留：同一会话不同车门 SNR 不同，这是该数据的声学变化，不是简单重复。四个通道共享近似相同的会话级真值，样本并不独立；解读远场数字时把它当成「四麦克风条件的池化」，不要当成四倍无关联小时数。完整收录表见第 2 节。

VoxConverse 同时跑 `dev` 和 `test`：`test` 用来对照 wiki；`dev` 用来搜阈值且单独报表，避免在 test 上调阈值。

### 5.4 确认采样率后拒绝重采样

抽查 AISHELL-5 与 VoxConverse 的 wav 均为 16 kHz 单声道 PCM16。脚本若读到其他采样率直接报错，而不是悄悄 `resample`。原因：静默重采样会引入额外变量；这批数据不需要这一步。多声道则先对声道取平均（本批实际都是单声道）。

不做峰值归一化：官方 `read_audio` 也只把 PCM 转成 `[-1,1]` 量级的线性波形，训练分布覆盖多种电平。额外归一化会改变与官方数字的可比性。

### 5.5 索引文件：有音频、有标签、能对齐才进评估

AISHELL-5：递归收集 wav，必须存在同 stem 的 `.TextGrid`，再按文件名分近场/远场。  
VoxConverse：以 rttm 为主表去找同名 wav，丢掉 `._*`。

这样不会因为缺标注或资源叉文件让某条音频「无标签却被当成全静音」。

### 5.6 推理：按官方流式契约逐步跑，文件之间重置状态

16 kHz 下 Silero 窗口固定 512，TEN 固定 256。尾块补零。Silero 与 `OnnxWrapper.audio_forward` / `get_speech_timestamps` 一致；TEN 官方 `examples/test.py` 用 `N // 256` 丢掉尾部，本评测为对齐「最后一窗也计分」而补零。

**v4**

- 输入当前窗 512 点，不拼 context。
- 状态 `h,c` 形状 `[2, 1, 64]`，输出 `hn,cn` 回写。
- `sr = 16000`。

**v5 / v6**

- 输入 `[context 64 | 当前窗 512]`，共 576 点。
- 第一步 context 全零；之后用上一输入的末 64 点。
- 状态 `state` 形状 `[2, 1, 128]`，输出 `stateN` 回写。
- `sr = 16000`。

**ten**

- 输入当前 hop **256** 点 **int16** PCM，不拼 context。
- 每个文件 `ten_vad_create(hop=256, threshold=0.5)` 得到新 handle；DLL 内部状态只在该文件内串联。
- 取 `out_probability`（0～1）进指标。DLL 的 `out_flag` 只是 `probability >= 0.5`，本评测自己按 0.5 切，不用这个 flag。
- 换文件时销毁 handle，避免跨文件泄漏。

换文件时 Silero 的状态与 context 清零。原因：VAD 的循环状态携带历史；跨文件不重置会把上一条的说话状态泄漏到下一条。

每个进程只开 1 个 ORT 线程（TEN 则每文件一个 DLL handle），用多进程按文件并行。RTF 只统计推理时间，不含解码与解析标注。

### 5.7 指标：先 AUC，再默认 0.5，最后才是「dev 上搜阈值」

| 指标 | 定义 | 为什么报它 |
| --- | --- | --- |
| ROC-AUC | 以概率排序，对 TPR–FPR 折线求面积 | wiki 主表；与阈值无关，适合三个模型对比 |
| Accuracy @ 0.5 | 正确帧 / 总帧 | wiki 的 Accuracy 表；0.5 是官方默认阈值 |
| Precision / Recall / F1 | 标准二分类 | 看「宁可不说」还是「宁可多检」 |
| FAR | FP / (FP+TN)，即非语音上的虚警率 | 噪声段是否被当成说话 |
| MR | FN / (FN+TP)，即语音上的漏检率 | 与 Recall 互补 |
| speech_ratio | 真值语音时长 / 音频时长 | 解释 Accuracy/F1 为什么会被类别不平衡拉动 |
| RTF | 推理秒 / 音频秒 | 质量对比之外的速度参考 |

帧全部拼接后一次计算（微平均）。宏平均会让极短文件和一小时节目权重相同；VAD 质量应按时间（帧）计。

**不要把「dev 上最大 F1 的阈值」当成主结论。** 远场语音占比约 79%，VoxConverse 约 90%。F1 在这种不平衡下会偏向很低的阈值（脚本网格是 0.05–0.95，步长 0.05，远场/Vox 常落到 0.05），虚警会很高。该数字只写在 JSON 的 `best_f1_on_dev` / `test_at_dev_best_t` 里，用于说明「若有人按 F1 调阈值会发生什么」，**不应当成推荐工作点**。对比模型请看 AUC 和 0.5。

这与 wiki 的精神一致：他们也是在独立的多域验证集上选阈值，再拿到各测试集上算 Accuracy；本脚本没有那份私有多域验证集，所以固定 0.5，并把「在本数据集 dev 上搜 F1」降为附录性质。

### 5.8 报表结构

每个模型输出三层：

1. `by_split`：如 `aishell5_far/eval1`、`voxconverse/test`，避免把 Dev 和 Eval 混成一个数。
2. `by_dataset`：近场全部 / 远场全部 / VoxConverse 全部。
3. `overall`：三个数据条件再池化。overall **不是**公平的「综合分」，因为远场小时数多、Vox 语音占比极高；看分条件表。

最后打印 `AUC / Acc@0.5 / F1@0.5` 对照表，便于 v4 / v5 / v6 / ten 并排。只跑 `--models ten` 时会读入已有 JSON 再合并写入。

### 5.9 刻意没有做的事

| 未做 | 原因 |
| --- | --- |
| 用 AISHELL-4 替换 AISHELL-5 | 本地数据是 AISHELL-5；wiki 的 AISHELL-4 数字不可直接当本表基线 |
| 能量门控远场真值 | 会改变任务定义；本协议明确测「会话级有人说话」 |
| 只取每个会话一个远场通道 | 会丢掉座位–麦克风匹配差异；代价是通道间相关 |
| 评估 `train/` | 不是测试划分 |
| collar / 分段后处理 | 见 5.1、4.4 |
| 8 kHz | 本批数据是 16 kHz；wiki 主表也是 16 kHz |
| GPU | 模型很小，CPU ORT 已是数百倍实时，排除设备差异 |

---

## 6. 一次完整数据流（单文件）

以 `D:\AISHELL-5-Data\Dev\001\DX01C01.wav` 为例：

1. 归入 `aishell5_far` / `dev`（文件名是 `DX01C01` 且存在 `DX01C01.TextGrid`）。
2. 读 TextGrid 全部 tier：非空 `text` 的 `[xmin, xmax]` 收集起来，重叠合并，得到会话级语音区间。
3. `soundfile` 读 wav。Silero → `float32`；TEN → `int16`。长度 N sample。帧数 `F = ceil(N / H)`，H 为该模型 hop。
4. 对 i = 0 … F−1：按第 4.4 节写 `labels[i] ∈ {0,1}`。
5. 按第 5.6 节流式推理，得到 `probs[i] ∈ [0,1]`。v4 不拼 context；v5/v6 拼 64 点 context；TEN 喂 256 点 int16。
6. 该文件的 `(labels, probs)` 进入对应 split 的大拼接数组。
7. 该 split 上算 AUC 与混淆矩阵（阈值 0.5）。

VoxConverse 的第 2 步换成解析 RTTM 的 `SPEAKER` 行并求并集，其余相同。

---

## 7. 如何复现

```text
python examples/eval_vad_datasets.py
python examples/eval_vad_datasets.py --models ten
python examples/eval_vad_datasets.py --limit 4 --workers 4
```

默认读取 `D:\AISHELL-5-Data` 与 `D:\voxconverse`，写出 `examples/eval_vad_results.json`。只跑 `--models ten` 时会**合并**进已有 JSON，保留 v4/v5/v6。TEN 需要 `D:\ten-vad\include\ten_vad.py` 与 `D:\ten-vad\lib\Windows\x64\ten_vad.dll`。换机器时改脚本顶部的 `AISHELL_ROOT` / `VOX_ROOT` / `TEN_ROOT`。`--limit` 按**每个数据条件**截断文件数，只用于冒烟，不能当正式分。
