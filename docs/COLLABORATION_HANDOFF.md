# Collaboration Handoff

Last updated: 2026-06-02, Asia/Shanghai.

This document captures the current working method, environment topology, and project state for future collaborators. The canonical agent rules remain in `AGENTS.md`; this file is a practical handoff and operating guide.

## Current State

The repo is on `main` and the latest pushed commit is:

```text
7af67b7 add source registry validation workflow
```

The v0/v1 data pipeline already has a working path:

```text
raw parquet
-> manifest / schema probe
-> proxy JSONL sample
-> token stats
-> packed uint32 token bin
-> packed dataset sanity check
-> tiny GPT smoke train
-> checkpoint save/resume
-> Docker training run
```

Recently completed:

- Source Registry v1 config directory:
  - `configs/sources/openwebmath.yaml`
  - `configs/sources/fineweb_edu_sample_350BT.yaml`
  - `configs/sources/fineweb2_cmn_Hani.yaml`
  - `configs/sources/starcoderdata.yaml`
- Source Registry loader and validator:
  - `pipeline/sources/registry.py`
  - `scripts/validate_source_registry_v1.py`
- Fresh-clone remote validation helper:
  - `scripts/remote_validate_from_git_v1.sh`
- Manifest v1 builder:
  - `scripts/build_manifest_v1.py`
  - output: `data/manifests/source_registry_v1/`
  - report: `data/reports/manifest_v1.md`
- Remote validation was run from a fresh clone on `guangzhoujump` and passed with:

```text
sources: 4
check_paths: true
missing_roots: []
ok: true
```

## Environments

Local machine:

- Use for code development, commits, and pushes.
- Do not assume local access to raw data, tokenizer files, shared storage, or GPUs.

Data processing machine:

```bash
ssh guangzhoujump "cmd"
```

- Shared repo root: `/mnt/kai_kpfs/weilai/train/pt-data-pipeline`
- Raw data root: `/mnt/kai_kpfs/weilai/dataset/raw`
- Tokenizer path: `/mnt/kai_kpfs/vol3/GLM-5.1-FP8`
- Prepared venv: `/mnt/kai_kpfs/weilai/train/pt-data-pipeline/.venv`
- Use for raw data inspection, schema probing, JSONL sampling, cleaning, token stats, packing, reports, and lightweight sanity checks.

GPU training machine:

```bash
ssh gzjp 'ssh 110.0.2.211 cmd'
```

- Use Docker image: `pt-pipeline:train-v0`
- Use for GPU training only.
- Prefer writing a shell script to shared storage, then invoking it remotely, to avoid nested SSH quoting issues.

## Standard Workflow

For code changes:

```bash
git pull
# edit code
python3 scripts/validate_source_registry_v1.py --config-dir configs/sources
python3 -m compileall pipeline scripts
git add ...
git commit -m "..."
git push
```

For changes that need shared storage, raw data, the tokenizer, or the data-machine venv, validate from a fresh remote clone after pushing:

```bash
bash scripts/remote_validate_from_git_v1.sh --cleanup -- \
  python3 scripts/validate_source_registry_v1.py \
    --config-dir configs/sources \
    --check-paths
```

Why this matters:

- The fresh clone proves pushed code is sufficient.
- It avoids accidentally relying on uncommitted local files.
- It avoids relying on the mutable shared working tree at `/mnt/kai_kpfs/weilai/train/pt-data-pipeline`.

Useful helper options:

```bash
bash scripts/remote_validate_from_git_v1.sh --help
```

Common variants:

```bash
# Keep the remote clone for debugging.
bash scripts/remote_validate_from_git_v1.sh --run-id debug_source_registry -- \
  python3 scripts/validate_source_registry_v1.py --config-dir configs/sources --check-paths

# Validate a specific pushed branch.
bash scripts/remote_validate_from_git_v1.sh --branch main --cleanup -- \
  python3 scripts/validate_source_registry_v1.py --config-dir configs/sources --check-paths
```

## Generated Artifacts Policy

Never commit generated or sensitive artifacts, including:

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
raw data
model weights
tokenizer files
tokens or credentials
```

Use versioned output names for all data-processing and training artifacts. Do not overwrite existing v0/v1 artifacts.

Good examples:

```text
proxy_train_starcoder_clean_basic_v1.jsonl
packed_glm_seq2048_starcoder_clean_basic_v1
smoke_gpt_4l384_starcoder_clean_basic_v1
```

## Current Important Artifacts

On the shared data-processing machine, existing important artifacts include:

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

Known current metrics:

```text
starcoder code probe:
  train docs: 20,000
  val docs: 2,000
  train packed seq_len=2048 sequences: 15,930
  val packed seq_len=2048 sequences: 1,564

packed eval buckets:
  val_starcoder_code: 1414 sequences
  val_starcoder_docs: 150 sequences
  val_webmath_chinese_web: 85 sequences
  val_webmath_english_web: 76 sequences
  val_webmath_math: 108 sequences
```

## Source Registry v1

The Source Registry is currently a readable, validated YAML config directory, not a database. Its job is to pin down how each source is read, interpreted, cleaned, sampled, split, and assigned to buckets.

Validator:

```bash
python3 scripts/validate_source_registry_v1.py --config-dir configs/sources
```

Remote path check:

```bash
bash scripts/remote_validate_from_git_v1.sh --cleanup -- \
  python3 scripts/validate_source_registry_v1.py \
    --config-dir configs/sources \
    --check-paths
```

First version scope:

- Config plus validation only.
- Existing sampling scripts are not yet migrated to read the registry.
- `starcoderdata` is the source id for StarCoderData.
- `raw/bigcode/the-stack-v2` is not included in this registry milestone.

## Manifest v1

Manifest v1 is the structured file-level index that sits between Source Registry and sampling/cleaning. It records what raw files exist, how they were detected, whether they are trainable, and which bucket they belong to. It does not sample, clean, tokenize, pack, or read text samples.

Build command on the data processing machine:

```bash
cd /mnt/kai_kpfs/weilai/train/pt-data-pipeline
source .venv/bin/activate

python3 scripts/build_manifest_v1.py --config-dir configs/sources
```

Lightweight smoke command for fresh-clone validation:

```bash
python3 scripts/build_manifest_v1.py \
  --config-dir configs/sources \
  --max-files-per-source 5
```

Outputs are generated artifacts and must not be committed:

```text
data/manifests/source_registry_v1/all_sources.jsonl
data/manifests/source_registry_v1/{source_id}.jsonl
data/reports/manifest_v1.md
```

Each manifest row uses `status` for probe state and `trainable` plus `trainable_reason` for whether downstream sample/clean scripts can consume the file. Current manifests remain flat-bucket only; there is no `subbucket` field in v1.

## Next Priorities

Recommended order:

1. Run Manifest v1 on the data processing machine and inspect `data/reports/manifest_v1.md`.
2. Add `eval_packed_buckets_v1.py`.
3. Evaluate existing code50 and webmath50 checkpoints across all packed eval buckets.
4. Add `clean_jsonl_v1.py`.
5. Produce `starcoder_clean_basic_v1` and `starcoder_clean_code_only_v1`.
6. Run token stats and packing for cleaned variants.
7. Run same-budget proxy training: raw vs clean.
8. Compare retained docs/tokens, reject reasons, mixed val loss, and bucket-wise val losses.

Do not spend time on large-scale training, multi-node training, or complex near-dedup before the cleaning plus bucket-wise evaluation loop is complete.
