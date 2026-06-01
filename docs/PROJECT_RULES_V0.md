# PT Data Pipeline v0 规约

## 0. 当前阶段定位

本项目当前阶段不是正式大规模预训练，不以产出可用模型为目标。

当前阶段目标是：

1. 在两天 GPU 窗口内最大化有效实验吞吐。
2. 建立可复用的数据、配置、训练、日志、评测闭环。
3. 用小模型 proxy 实验为下一轮正式训练提供依据。
4. 避免从零构建复杂训练系统、复杂数据平台或大型分布式框架。
5. 所有结果必须可复现、可比较、可追溯。

本阶段允许训练出的模型质量很差，但不允许实验不可复现、数据不可追踪、配置不可回放。

---

## 1. 强制路径规约

所有项目代码、配置、脚本、报告、实验记录必须位于：

```bash
/mnt/kai_kpfs/weilai/train/pt-data-pipeline
```

禁止在项目根目录外开发项目代码。

唯一允许读取的原始数据根目录为：

```bash
/mnt/kai_kpfs/weilai/dataset/raw
```

当前已知数据源包括：

```bash
/mnt/kai_kpfs/weilai/dataset/raw/bigcode
/mnt/kai_kpfs/weilai/dataset/raw/fineweb2_cmn_Hani
/mnt/kai_kpfs/weilai/dataset/raw/fineweb_edu_sample_350BT
/mnt/kai_kpfs/weilai/dataset/raw/openwebmath
```

所有中间产物必须写入项目目录下：

```bash
/mnt/kai_kpfs/weilai/train/pt-data-pipeline/data
```

所有 checkpoint 必须写入：

```bash
/mnt/kai_kpfs/weilai/train/pt-data-pipeline/models/checkpoints
```

---

## 2. 环境规约

所有命令默认从项目根目录执行：

```bash
cd /mnt/kai_kpfs/weilai/train/pt-data-pipeline
source /mnt/kai_kpfs/weilai/train/pt-data-pipeline/.venv/bin/activate
```

任何脚本、README、实验记录中不得省略虚拟环境激活命令。

当前 Python 环境必须至少满足：

```python
import pyarrow
import pandas
import tqdm
```

后续新增依赖必须写入：

```bash
requirements.txt
```

开发、测试、lint 依赖写入：

```bash
requirements-dev.txt
```

---

## 3. 今日训练框架选择

当前阶段优先使用 LitGPT 或同等级极简训练框架。

选择理由：

1. 代码简单，便于读懂和快速修改。
2. 训练、数据、模型配置已有基本结构。
3. 能较快跑通 scratch pretrain / continual pretrain。
4. 适合小模型 proxy 实验。
5. 不把时间消耗在 Megatron 级别的分布式复杂度上。

当前阶段不追求：

1. 极致 MFU。
2. 完整 Megatron 并行策略。
3. 大模型长期稳定训练。
4. 复杂 pipeline parallel / tensor parallel。
5. 完整数据治理系统。

正式训练阶段可以重新评估 Megatron-LM、NeMo、TorchTitan 或自研训练栈。

---

## 4. GPU 使用原则

本轮 GPU 机会稀缺，原则是：

1. GPU 不空转。
2. 任何实验必须能产生下一轮可复用信息。
3. 不做不可比较的随机实验。
4. 优先跑 proxy，而不是追求大模型结果。
5. 宁可跑多个小实验，也不要押注一个大实验。

实验优先级：

1. 数据吞吐测试。
2. tokenization + packing 正确性测试。
3. 训练框架 smoke test。
4. 小模型 loss curve。
5. batch size / learning rate / sequence length proxy。
6. 数据 mixture proxy。
7. 模型结构 proxy。
8. 多机扩展性测试。

---

## 5. 本阶段数据规约

当前不建设复杂数据清洗平台，只做最小可用数据闭环。

### 5.1 数据源分桶

必须至少定义以下 bucket：

```text
code
math
english_web
chinese_web
agent_like
unknown
```

当前数据映射建议：

```text
bigcode                     -> code
openwebmath                 -> math
fineweb_edu_sample_350BT    -> english_web
fineweb2_cmn_Hani           -> chinese_web
```

如果没有明确 agent 数据，则暂不伪造 agent 数据；可以预留 bucket，但不强行填充。

### 5.2 Manifest 优先

任何训练不得直接读取 raw dataset。

训练前必须生成 manifest：

```bash
data/manifests/*.jsonl
```

每一行至少包含：

```json
{
  "source": "bigcode",
  "bucket": "code",
  "path": "...",
  "format": "parquet|jsonl|txt|arrow",
  "num_bytes": 123,
  "num_rows": 456,
  "status": "ok|skipped|error"
}
```

如果已经能统计 token，则增加：

```json
{
  "estimated_tokens": 123456
}
```

### 5.3 最小过滤

当前只做硬过滤：

1. 空文本过滤。
2. 极短文本过滤。
3. 极长单样本过滤。
4. 非 UTF-8 或解码失败过滤。
5. 明显二进制内容过滤。
6. 明显 secret/private key/API key 过滤。
7. 重复 path 或重复 exact text hash 过滤。

当前不强求：

1. 复杂 near dedup。
2. 高级质量分类器。
3. OCR 质量模型。
4. 完整 PII 检测。
5. 完整 license 判定。
6. eval contamination 完整检测。

这些进入下一轮工程任务。

---

## 6. Tokenization 规约

当前阶段优先复用目标 base model tokenizer。

如果目标不明确，默认优先使用 Qwen/Llama 类 tokenizer，而不是今天自训 tokenizer。

原因：

1. 两天内自训 tokenizer 收益不稳定。
2. 代码和中英文混合场景下 tokenizer 选择会显著影响结果。
3. 使用成熟 tokenizer 更利于和开源模型继续预训练对齐。
4. proxy 阶段更重要的是比较数据、batch、模型结构趋势。

所有 tokenized 数据必须写入：

```bash
data/stage/tokenized
```

所有 packed sequence 必须写入：

```bash
data/stage/sequences
```

训练只能读取 packed sequence 或 LitGPT 可直接消费的 processed dataset。

---

## 7. 实验配置规约

任何实验必须有独立配置文件，禁止只靠命令行临时参数。

配置文件建议位于：

```bash
configs/experiments
```

命名格式：

```text
YYYYMMDD_<stage>_<model>_<data>_<purpose>.yaml
```

例如：

```text
20260530_proxy_120m_code_lr_sweep.yaml
20260530_proxy_300m_mix_ablation.yaml
20260531_scale_300m_32node_throughput.yaml
```

每个配置必须包含：

```yaml
run:
  name:
  seed:
  framework:
  git_commit:
  notes:

data:
  mixture:
  train_manifest:
  val_manifest:
  seq_len:
  tokenizer:
  num_tokens_target:

model:
  family:
  n_layer:
  n_head:
  n_embd:
  vocab_size:
  params_estimate:

optim:
  optimizer:
  lr:
  min_lr:
  weight_decay:
  beta1:
  beta2:
  grad_clip:
  warmup_tokens:
  total_tokens:

train:
  global_batch_tokens:
  micro_batch_size:
  grad_accum_steps:
  precision:
  compile:
  checkpoint_interval:
  eval_interval:
  log_interval:

system:
  nodes:
  gpus_per_node:
  launcher:
  docker_image:
```

---

## 8. Proxy 模型规约

本阶段不直接训练大模型。

建议 proxy 模型规模：

```text
tiny     30M - 60M
small    100M - 150M
medium   250M - 400M
```

第一天必须先跑：

1. tiny 单卡。
2. small 单机多卡。
3. medium 多机短跑。

只有当前一级稳定后，才能进入下一级。

---

## 9. 本轮核心实验矩阵

本轮最多做 6 类实验，禁止无限扩散。

### 9.1 Smoke Test

目标：验证训练代码、数据读取、loss、checkpoint、resume。

最低要求：

1. loss 不 NaN。
2. loss 有下降趋势。
3. checkpoint 可保存。
4. resume 后 step 连续。
5. 日志完整。

### 9.2 数据 Mixture Proxy

比较：

```text
code-only
code + math
code + english_web
code + chinese_web
code + math + english_web + chinese_web
```

目标不是效果绝对值，而是 loss 曲线、val loss、训练稳定性、吞吐差异。

### 9.3 Sequence Length Proxy

比较：

```text
seq_len = 1024
seq_len = 2048
seq_len = 4096
```

如果显存紧张，优先 1024/2048。

### 9.4 Batch / LR Proxy

对 small 或 medium 模型做少量学习率比较：

```text
lr = 1e-3
lr = 6e-4
lr = 3e-4
```

记录 loss 曲线、是否发散、warmup 是否足够。

### 9.5 模型结构 Proxy

只比较少量结构，不做大搜索。

可比较：

```text
deep-narrow
balanced
shallow-wide
```

所有模型总参数量尽量接近。

### 9.6 多机吞吐 Proxy

目标不是长期训练，而是知道 32 台机器是否值得用于下一轮正式训练。

必须记录：

1. tokens/sec。
2. samples/sec。
3. GPU 利用率。
4. dataloader wait。
5. checkpoint 写入耗时。
6. 节点间失败率。
7. resume 成功率。

---

## 10. μP 规约

本阶段可以为 μP 做准备，但不强制今天完成完整 μP 实现。

今天必须记录以下信息：

1. 模型宽度。
2. 深度。
3. head 数。
4. hidden size。
5. batch tokens。
6. learning rate。
7. warmup tokens。
8. optimizer 参数。
9. loss 曲线。
10. 是否发散。

如果训练框架不原生支持 μP，不允许为了 μP 重写训练框架而耽误 GPU 使用。

本阶段 μP 的合理目标是：

1. 建立 width-scaling 实验表。
2. 固定 depth，比较不同 width 下的稳定 learning rate。
3. 为下一轮正式 μP parameterization 实现提供数据。
4. 不把 μP 变成今天的主线阻塞项。

---

## 11. 日志和报告规约

每个实验必须生成一个 run directory：

```bash
data/reports/runs/<run_name>
```

至少包含：

```text
config.yaml
stdout.log
stderr.log
metrics.jsonl
env.txt
git.txt
data_manifest_snapshot.jsonl
notes.md
```

metrics.jsonl 每行至少包含：

```json
{
  "step": 100,
  "tokens": 2048000,
  "train_loss": 4.2,
  "val_loss": 4.4,
  "lr": 0.0006,
  "tokens_per_sec": 12345,
  "gpu_mem_gb": 72.1
}
```

两天结束时必须产出：

```bash
data/reports/summary_48h.md
```

内容包括：

1. 哪些数据成功进入训练。
2. 每个 bucket 的样本数、bytes、估算 tokens。
3. 哪些实验跑通。
4. 哪些实验失败。
5. 最稳定的配置。
6. 最有希望的数据 mixture。
7. 下一轮正式训练建议。
8. 当前最大工程风险。

---

## 12. 验收标准

两天结束不以模型效果验收，而以工程闭环验收。

必须完成：

1. 环境启动文档。
2. 数据 manifest 生成。
3. 至少一个 tokenization / packing 流程。
4. 至少一个 tiny 模型训练成功。
5. 至少一个 small 或 medium proxy 实验成功。
6. 至少一个 checkpoint resume 成功。
7. 至少一个 val loss 评估成功。
8. 至少一份 48h summary 报告。
9. 至少一份下一轮正式训练建议。

加分项：

1. 32 机短跑吞吐测试。
2. 数据 mixture 对比。
3. seq_len 对比。
4. lr 对比。
5. μP 准备性 width-scaling 实验。
6. Eval contamination 初筛。
7. Secrets 过滤统计。

---

## 13. 禁止事项

本阶段禁止：

1. 直接读取 raw 数据训练。
2. 没有 config 的实验。
3. 没有日志的实验。
4. 没有 checkpoint resume 测试的长跑。
5. 一上来跑大模型。
6. 为了完美数据清洗阻塞训练。
7. 为了完美训练框架阻塞训练。
8. 为了 Megatron 工程阻塞今天的 proxy。
9. 无 eval、无 manifest、无 config 的 GPU 消耗。
10. 在项目根目录外写项目代码。

---

## 14. 当前阶段最终判断

今天应该优先使用 LitGPT 或同类极简框架启动 proxy pretraining。

正式训练周期再评估 Megatron/NeMo/TorchTitan。

当前两天最重要的产出不是模型，而是：

1. 可复用数据配置。
2. 可复现实验配置。
3. 可追踪日志。
4. 可恢复训练入口。
5. 小模型 proxy 结果。
6. 下一轮正式训练决策依据。
