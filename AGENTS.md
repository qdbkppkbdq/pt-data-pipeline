# AGENT.md

This repository is a temporary but actively used development workspace for a code / agent-oriented LLM pretraining data pipeline. The goal is not to train a production model here. The goal is to build, validate, and document a reproducible pipeline for:

* raw data inspection
* schema probing
* JSONL sample generation
* data cleaning
* token statistics
* token packing
* tiny/proxy GPT training
* checkpoint/resume smoke tests
* bucket-wise evaluation
* small ablation experiments for data quality and mixture decisions

## Current project status

The repository already has a working v0/v1 pipeline:

```text
raw parquet
→ manifest / schema probe
→ proxy JSONL sample
→ token stats
→ packed uint32 token bin
→ packed dataset sanity check
→ tiny GPT smoke train
→ checkpoint save/resume
→ Docker training run
```

Current major completed items:

* web/math v0 pipeline is working
* StarCoderData code probe is working
* token stats for code probe are working
* packing for code probe is working
* packed sanity check is passing
* Docker-based tiny GPT training is working
* 3-step and 50-step proxy runs have been run
* eval buckets have been split and packed
* git baseline has been created and pushed to GitHub

Current important artifacts:

```text
data/stage/proxy/proxy_train_v0.jsonl
data/stage/proxy/proxy_val_v0.jsonl
data/stage/packed_glm_seq2048/

data/stage/proxy/proxy_train_starcoder_code_probe_v1.jsonl
data/stage/proxy/proxy_val_starcoder_code_probe_v1.jsonl
data/stage/packed_glm_seq2048_starcoder_code_probe_v1/

data/stage/packed_eval_seq2048_v1/
data/reports/runs/smoke_gpt_4l384_starcoder_code_probe_v1/
data/reports/runs/smoke_gpt_4l384_webmath_v0/
```

Important current metrics:

```text
starcoder code probe:
  train docs: 20,000
  val docs: 2,000
  train packed seq_len=2048 sequences: 15,930
  val packed seq_len=2048 sequences: 1,564

50-step proxy runs:
  starcoder code probe final val_loss: about 8.346
  web/math v0 final val_loss: about 8.346

packed eval buckets:
  val_starcoder_code: 1414 sequences
  val_starcoder_docs: 150 sequences
  val_webmath_chinese_web: 85 sequences
  val_webmath_english_web: 76 sequences
  val_webmath_math: 108 sequences
```

The next major priority is data cleaning plus bucket-wise evaluation, not more random smoke runs.

## Development environment topology

There are three working environments.

### 1. Local machine

Use the local machine for normal code development.

Typical workflow:

```bash
git clone https://github.com/qdbkppkbdq/pt-data-pipeline.git
cd pt-data-pipeline

# edit code locally
git add ...
git commit -m "..."
git push
```

Do not assume local machine has access to raw datasets, shared storage, GPUs, or the tokenizer path.

### 2. Data processing machine

The data processing machine is used for CPU/data pipeline work.

Access:

```bash
ssh guangzhoujump "cmd"
```

This machine has access to the shared storage and a project `.venv` that is already prepared for data processing tasks.

Typical project root on shared storage:

```bash
/mnt/kai_kpfs/weilai/train/pt-data-pipeline
```

Typical raw data root:

```bash
/mnt/kai_kpfs/weilai/dataset/raw
```

Typical tokenizer path:

```bash
/mnt/kai_kpfs/vol3/GLM-5.1-FP8
```

Typical usage:

```bash
ssh guangzhoujump "cd /mnt/kai_kpfs/weilai/train/pt-data-pipeline && git pull"

ssh guangzhoujump "cd /mnt/kai_kpfs/weilai/train/pt-data-pipeline && source .venv/bin/activate && python3 scripts/token_stats_v0.py --help"
```

Use this machine for:

* scanning raw data
* schema probe
* JSONL sampling
* data cleaning
* token statistics
* token packing
* report generation
* lightweight sanity checks

Do not use this machine for GPU training unless explicitly instructed.

### 3. Training GPU machine

The GPU training server is used for Docker-based training runs.

Access:

```bash
ssh gzjp 'ssh 110.0.2.211 cmd'
```

The GPU server already has the Docker image:

```text
pt-pipeline:train-v0
```

Use this image for training. Do not rely on the host Python environment for GPU training.

Typical Docker command shape:

```bash
docker run --rm \
  --gpus all \
  --network host \
  --privileged \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e PYTHONUNBUFFERED=1 \
  -e TOKENIZERS_PARALLELISM=false \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -v /mnt/kai_kpfs/weilai/train/pt-data-pipeline:/mnt/kai_kpfs/weilai/train/pt-data-pipeline \
  -v /mnt/kai_kpfs/weilai/dataset:/mnt/kai_kpfs/weilai/dataset \
  -v /mnt/kai_kpfs/vol3/GLM-5.1-FP8:/mnt/kai_kpfs/vol3/GLM-5.1-FP8 \
  -w /mnt/kai_kpfs/weilai/train/pt-data-pipeline \
  pt-pipeline:train-v0 \
  bash -lc 'python scripts/train_tiny_gpt_smoke_v0.py --help'
```

The jump host and GPU server share storage under:

```bash
/mnt/kai_kpfs/weilai/train
```

To avoid nested SSH quote escaping problems, prefer writing a shell script to shared storage from the data machine or local machine, then run it on the GPU server.

Example:

```bash
# Copy or create a script on shared storage:
scp run_train.sh guangzhoujump:/mnt/kai_kpfs/weilai/train/run_train.sh

# Run it on the GPU machine:
ssh gzjp 'ssh 110.0.2.211 bash /mnt/kai_kpfs/weilai/train/run_train.sh'
```

This pattern is preferred for non-trivial Docker commands.

## Hard rules for agents

Do not commit generated artifacts.

Never commit:

```text
.venv/
data/stage/
data/reports/
data/manifests/
models/checkpoints/
*.bin
*.pt
*.jsonl
.env
.git-credentials
raw data
model weights
tokenizer files
GitHub tokens
Hugging Face tokens
SSH keys
```

Do not overwrite existing v0/v1 artifacts. Use versioned names.

Good examples:

```text
proxy_train_starcoder_clean_basic_v1.jsonl
packed_glm_seq2048_starcoder_clean_basic_v1
smoke_gpt_4l384_starcoder_clean_basic_v1
```

Bad examples:

```text
proxy_train_v0.jsonl
packed_glm_seq2048
smoke_tiny_gpt_v0
```

Do not draw conclusions from mixed validation loss alone. Cleaning and mixture decisions require bucket-wise evaluation.

GPU training must run inside Docker on the GPU server.

Host `.venv` is for data processing only.

## Current priority

The current priority is:

```text
1. Implement bucket-wise eval for existing checkpoints.
2. Implement clean_jsonl_v1.py.
3. Generate clean_basic and clean_code_only data variants.
4. Run token stats and packing for cleaned variants.
5. Run same-budget proxy training: raw vs clean.
6. Compare bucket-wise losses and retention/reject reports.
```

Do not spend time on large-scale training, multi-node training, or complex near-dedup before the above items are complete.

## Expected data schema

JSONL records should follow this shape:

```json
{
  "id": "starcoderdata:python:...",
  "source": "starcoderdata",
  "bucket": "code",
  "language": "python",
  "text": "...",
  "meta": {
    "raw_file": "...",
    "row_group": 0,
    "raw_id": "...",
    "repo_path": "...",
    "repo_name": "...",
    "stars": 0,
    "content_sha1": "...",
    "norm_sha1": "...",
    "clean_version": "code_clean_v1"
  }
}
```

Required fields:

```text
id
source
bucket
text
```

Recommended fields:

```text
language
meta.raw_file
meta.repo_path
meta.repo_name
meta.content_sha1
meta.norm_sha1
meta.clean_version
```

Current bucket values include:

```text
code
docs
math
english_web
chinese_web
```

Future bucket values may include:

```text
markup_config
repo_activity
notebook
agent_like
unknown
```

## Standard data-processing commands

Run on the data processing machine:

```bash
ssh guangzhoujump "cd /mnt/kai_kpfs/weilai/train/pt-data-pipeline && source .venv/bin/activate && python3 scripts/token_stats_v0.py --help"
```

Token stats example:

```bash
cd /mnt/kai_kpfs/weilai/train/pt-data-pipeline
source .venv/bin/activate

python3 scripts/token_stats_v0.py \
  --tokenizer /mnt/kai_kpfs/vol3/GLM-5.1-FP8 \
  --train data/stage/proxy/proxy_train_starcoder_code_probe_v1.jsonl \
  --val data/stage/proxy/proxy_val_starcoder_code_probe_v1.jsonl \
  --seq-lens 1024,2048,4096 \
  --report data/reports/token_stats_starcoder_code_probe_v1.md \
  --json-out data/reports/token_stats_starcoder_code_probe_v1.json
```

Packing example:

```bash
cd /mnt/kai_kpfs/weilai/train/pt-data-pipeline
source .venv/bin/activate

python3 scripts/pack_tokens_v0.py \
  --tokenizer /mnt/kai_kpfs/vol3/GLM-5.1-FP8 \
  --seq-len 2048 \
  --train data/stage/proxy/proxy_train_starcoder_code_probe_v1.jsonl \
  --val data/stage/proxy/proxy_val_starcoder_code_probe_v1.jsonl \
  --out-dir data/stage/packed_glm_seq2048_starcoder_code_probe_v1
```

## Standard training command pattern

Prefer creating a shell script on shared storage, then invoking it remotely.

Example shared script path:

```bash
/mnt/kai_kpfs/weilai/train/run_train_code50.sh
```

Run it:

```bash
ssh gzjp 'ssh 110.0.2.211 bash /mnt/kai_kpfs/weilai/train/run_train_code50.sh'
```

Inside the script, use Docker:

```bash
cd /mnt/kai_kpfs/weilai/train/pt-data-pipeline

docker run --rm \
  --gpus all \
  --network host \
  --privileged \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e PYTHONUNBUFFERED=1 \
  -e TOKENIZERS_PARALLELISM=false \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -v /mnt/kai_kpfs/weilai/train/pt-data-pipeline:/mnt/kai_kpfs/weilai/train/pt-data-pipeline \
  -v /mnt/kai_kpfs/weilai/dataset:/mnt/kai_kpfs/weilai/dataset \
  -v /mnt/kai_kpfs/vol3/GLM-5.1-FP8:/mnt/kai_kpfs/vol3/GLM-5.1-FP8 \
  -w /mnt/kai_kpfs/weilai/train/pt-data-pipeline \
  pt-pipeline:train-v0 \
  bash -lc '
python scripts/train_tiny_gpt_smoke_v0.py \
  --packed-dir data/stage/packed_glm_seq2048_starcoder_code_probe_v1 \
  --out-dir models/checkpoints/smoke_gpt_4l384_starcoder_code_probe_v1 \
  --run-dir data/reports/runs/smoke_gpt_4l384_starcoder_code_probe_v1 \
  --batch-size 2 \
  --grad-accum 4 \
  --max-steps 50 \
  --eval-interval 10 \
  --eval-batches 8 \
  --n-layer 4 \
  --n-head 6 \
  --n-embd 384 \
  --num-workers 2
'
```

## Development workflow

Local machine:

```bash
git pull
# edit code
git add ...
git commit -m "..."
git push
```

Standard remote validation workflow:

```bash
# 1. Commit and push local code first.
git push

# 2. Validate from a fresh remote clone on the data processing machine.
bash scripts/remote_validate_from_git_v1.sh -- \
  python3 scripts/validate_source_registry_v1.py \
    --config-dir configs/sources \
    --check-paths
```

Use this workflow when a change needs access to shared storage, raw data,
the prepared tokenizer, or the data-machine `.venv`. The remote validation
must clone from git instead of reusing the mutable shared working tree, so the
test proves that pushed code is sufficient to reproduce the result.

Data processing machine:

```bash
ssh guangzhoujump "cd /mnt/kai_kpfs/weilai/train/pt-data-pipeline && git pull"
```

GPU machine:

```bash
ssh gzjp 'ssh 110.0.2.211 "cd /mnt/kai_kpfs/weilai/train/pt-data-pipeline && git pull"'
```

If a command contains multiple levels of quotes, write a script to shared storage and run the script remotely instead.

## What counts as done

For a data-processing change:

```text
- script runs successfully
- output paths are versioned
- report is generated
- reject/accept counts are visible if cleaning is involved
- no generated data is committed
```

For a training change:

```text
- Docker command works on GPU server
- run_dir contains metrics
- checkpoint is saved
- command/config is recorded
- no checkpoint is committed
```

For a cleaning change:

```text
- raw input and clean output are both preserved
- reject reasons are counted
- accepted/rejected sample report is produced
- token stats are produced after cleaning
- packing passes sanity check
- same-budget proxy comparison is possible
```

## Immediate next tasks

Recommended order:

```text
1. Add eval_packed_buckets_v1.py.
2. Evaluate existing code50 and webmath50 checkpoints across all packed eval buckets.
3. Add clean_jsonl_v1.py.
4. Produce:
   - starcoder_clean_basic_v1
   - starcoder_clean_code_only_v1
5. Run token stats and packing for cleaned variants.
6. Run raw vs clean proxy runs with identical model/training config.
7. Write a short summary report comparing:
   - retained docs/tokens
   - reject reasons
   - mixed val loss
   - bucket-wise val losses
```

## Security notes

This repository may be public. Never write tokens or credentials into tracked files.

If GitHub credentials were stored on a temporary machine, remove them when finished:

```bash
rm -f ~/.git-credentials
git config --global --unset credential.helper
```
