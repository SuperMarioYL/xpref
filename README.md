[English](./README.en.md) · [Website](https://xpref.lei6393.com) · [GitHub](https://github.com/SuperMarioYL/xpref)

<picture>
  <source media="(max-width: 600px) and (prefers-color-scheme: dark)" srcset="./assets/presentation/hero-mobile-dark.svg">
  <source media="(max-width: 600px)" srcset="./assets/presentation/hero-mobile-light.svg">
  <source media="(prefers-color-scheme: dark)" srcset="./assets/presentation/hero-dark.svg">
  <img src="./assets/presentation/hero-light.svg" width="960" alt="Hero diagram">
</picture>

# xpref

**测试预取策略前，先检查专家预测。**

xpref 结合最近路由 logits 与已观察专家转移，预测下一 token 的专家 ID，在保存轨迹上评估，并可向操作系统提示映射后的 checkpoint 字节区间。

## 为什么需要它

预取实验既需要预测，也需要专家到权重字节的准确映射。保存轨迹能先研究召回率，再接入引擎测量真实 I/O。

- **比较下一 token 集合** — 预测与后一 token 的集合评分。
- **独立检查字节映射** — 显式区间让布局假设可见。
- **区分召回与速度** — JSON 将投影吞吐与召回并列标注。

## 架构

<picture>
  <source media="(max-width: 600px) and (prefers-color-scheme: dark)" srcset="./assets/presentation/architecture-mobile-dark.svg">
  <source media="(max-width: 600px)" srcset="./assets/presentation/architecture-mobile-light.svg">
  <source media="(prefers-color-scheme: dark)" srcset="./assets/presentation/architecture-dark.svg">
  <img src="./assets/presentation/architecture-light.svg" width="960" alt="Architecture diagram">
</picture>

轨迹或 ring 记录提供逐层 logits 与已激活 ID。Predictor 混合 softmax 先验与衰减转移计数，选择与活跃集合等大的预测。ExpertLayout 将 ID 映射为区间，Prefetcher 使用 mmap 和 MADV_WILLNEED；评估将预测与下一 token 的真实集合比较。

| 组件 | 职责 |
| --- | --- |
| `Trace / ring input` | src/xpref/trace.py; ringbuf.py |
| `Expert predictor` | src/xpref/predictor.py |
| `Byte-range layout` | src/xpref/prefetch.py |
| `Advisory readahead` | mmap + MADV_WILLNEED |
| `Recall evaluation` | src/xpref/attach.py |

## 安装与快速上手

使用仓库清单声明的运行时版本。以下源码安装步骤可复现随仓示例。

```bash
git clone https://github.com/SuperMarioYL/xpref.git
cd xpref
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

支持 Python 3.10+，此处安装使用 3.12。示例评估完整 128-token 合成轨迹，并提示一个页对齐的合成文件区间。

```bash
.venv/bin/python examples/presentation_demo.py
```

## 实际运行示例

<picture>
  <source media="(max-width: 600px) and (prefers-color-scheme: dark)" srcset="./assets/presentation/process-mobile-dark.svg">
  <source media="(max-width: 600px)" srcset="./assets/presentation/process-mobile-light.svg">
  <source media="(prefers-color-scheme: dark)" srcset="./assets/presentation/process-dark.svg">
  <img src="./assets/presentation/process-light.svg" width="960" alt="Process diagram">
</picture>

The predictor reports recall on the synthetic trace; the file experiment reports only whether an advisory range hint was accepted.

```text
Synthetic trace evaluation; t/s fields are formula projections:
{
  "num_tokens": 128,
  "num_layers": 8,
  "num_experts": 896,
  "num_active": 16,
  "recall_at_k": 0.8153912401574803,
  "recall_at_16": 0.8153912401574803,
  "per_layer_recall": [
    0.8208661417322834,
    0.8095472440944882,
    0.8115157480314961,
    0.8188976377952756,
    0.8125,
    0.8154527559055118,
    0.8184055118110236,
    0.8159448818897638
  ],
  "predictions_made": 1016,
  "reactive_tps": 4.0,
  "projected_predictive_tps": 12.0,
  "speedup": 3.0
}
{"synthetic_checkpoint_bytes": 16384, "hint_api_available": true, "hinted_offset": 4096, "hinted_bytes": 4096}
```

完整命令与输出保存在 [docs/demo-results.json](./docs/demo-results.json). 输入和复现代码均随仓提供。

![已有脚本绘制示意](./assets/demo.gif)

已有 GIF 由 scripts/gen_demo_gif.py 使用预设文字绘制，是示意而非实时硬件录制。可复现结果以上方实际输出为准。

## 用法

安装后在仓库根目录运行以下命令；处理自己的数据时替换相应路径。

```bash
xpref trace-info --trace samples/k3-q4-128tok.bin
xpref eval --trace samples/k3-q4-128tok.bin --json
xpref eval --trace samples/k3-q4-128tok.bin --topk-weight 1 --ngram-weight 0
# After adapting and validating your own engine ring and tensor layout:
xpref attach --ring /path/to/router.xring --checkpoint /path/to/weights.gguf --layout /path/to/layout.json --max-tokens 128
```

## 配置

eval 支持 --ngram-n（3）、--topk-weight（0.6）、--ngram-weight（0.4）。attach 要求 --checkpoint，并选择 --replay 或 --ring；裸 ring 名在平台共享内存/临时目录解析。--layout 使用 JSON {layer: {expert_id: [offset, length]}}。patch-path 定位参考 patch 供检查；真实集成评估须独立测量引擎吞吐。

## 集成与职责分工

<picture>
  <source media="(max-width: 600px) and (prefers-color-scheme: dark)" srcset="./assets/presentation/integrations-mobile-dark.svg">
  <source media="(max-width: 600px)" srcset="./assets/presentation/integrations-mobile-light.svg">
  <source media="(prefers-color-scheme: dark)" srcset="./assets/presentation/integrations-dark.svg">
  <img src="./assets/presentation/integrations-light.svg" width="960" alt="Integrations diagram">
</picture>

根据工作流选择输入与输出路径。本文本地示例验证其中明确说明的子流程。

| 路径 | 已实现职责 |
| --- | --- |
| Binary router trace | Offline logits and fired IDs |
| Ring buffer | Prototype live record consumer |
| Explicit JSON layout | Expert-to-byte mapping |
| mmap / madvise | OS page-cache hints |
| CLI JSON | Recall and formula projections |

## 限制与后续方向

- 随仓带 K3 标签的轨迹由 scripts/gen_sample_trace.py 合成，其召回不代表实测模型准确率；4/12 t/s 是写入代码的投影假设，不是硬件实测。
- MADV_WILLNEED 只是建议；提示字节数不证明磁盘读取、驻留或吞吐提升。均匀布局只是合成近似，真实权重需核实字节区间。
- 随仓引擎 patch 是需要适配的示意参考，不是已验证的即用 llama.cpp 集成。示例不附加在线模型，也不向 GPU 内存搬运数据。

已验证的引擎生产端、准确 GGUF 布局提取与端到端性能实测是下一步集成工作；GPU/锁页内存搬运不在当前链路。

## 许可与贡献

许可见 [LICENSE](./LICENSE). 反馈问题时请提供最小输入、执行命令和实际输出。
