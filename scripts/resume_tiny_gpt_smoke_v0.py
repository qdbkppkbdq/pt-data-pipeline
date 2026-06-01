#!/usr/bin/env python3
import argparse
import json
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path("/mnt/kai_kpfs/weilai/train/pt-data-pipeline")
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train_tiny_gpt_smoke_v0 import (  # noqa: E402
    GPTConfig,
    PackedBinDataset,
    TinyGPT,
    evaluate,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--packed-dir", type=Path, default=PROJECT_ROOT / "data/stage/packed_glm_seq2048")
    parser.add_argument("--target-step", type=int, default=5)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--eval-batches", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "models/checkpoints/smoke_tiny_gpt_v0")
    parser.add_argument("--run-dir", type=Path, default=PROJECT_ROOT / "data/reports/runs/smoke_tiny_gpt_v0")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
    old_config = ckpt["config"]
    start_step = int(ckpt["step"])

    if args.target_step <= start_step:
        raise ValueError(f"target-step={args.target_step} must be > checkpoint step={start_step}")

    meta = json.loads((args.packed_dir / "meta.json").read_text())
    stored_len = int(meta["stored_len"])

    model_cfg = GPTConfig(**old_config["model"])
    model = TinyGPT(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(old_config["lr"]),
        betas=(0.9, 0.95),
        weight_decay=float(old_config["weight_decay"]),
    )
    optimizer.load_state_dict(ckpt["optimizer"])

    train_ds = PackedBinDataset(args.packed_dir / "train.bin", stored_len=stored_len)
    val_ds = PackedBinDataset(args.packed_dir / "val.bin", stored_len=stored_len)

    batch_size = int(old_config["batch_size"])
    grad_accum = int(old_config["grad_accum"])
    seq_len = int(old_config["seq_len"])

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.run_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = args.run_dir / "resume_metrics.jsonl"

    print("== resume config ==")
    print(json.dumps({
        "resume": str(args.resume),
        "checkpoint_step": start_step,
        "target_step": args.target_step,
        "device": str(device),
        "precision": str(precision),
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "seq_len": seq_len,
        "global_batch_tokens": batch_size * grad_accum * seq_len,
    }, indent=2, ensure_ascii=False))

    model.train()
    data_iter = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)

    t0 = time.time()
    last_time = t0

    for step in range(start_step + 1, args.target_step + 1):
        step_loss = 0.0

        for _ in range(grad_accum):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                x, y = next(data_iter)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.autocast(
                device_type="cuda",
                dtype=precision,
                enabled=device.type == "cuda",
            ):
                _, loss = model(x, y)
                loss = loss / grad_accum

            loss.backward()
            step_loss += float(loss.item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        now = time.time()
        dt = now - last_time
        last_time = now

        tokens_this_step = batch_size * grad_accum * seq_len

        record = {
            "step": step,
            "resumed_from": start_step,
            "train_loss": step_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "tokens_this_step": tokens_this_step,
            "tokens_per_sec": tokens_this_step / max(dt, 1e-9),
            "elapsed_sec": now - t0,
        }

        if torch.cuda.is_available():
            record["gpu_mem_allocated_gb"] = torch.cuda.memory_allocated() / 1024**3
            record["gpu_mem_reserved_gb"] = torch.cuda.memory_reserved() / 1024**3

        if step % args.eval_interval == 0 or step == args.target_step:
            record["val_loss"] = evaluate(
                model,
                val_loader,
                device=device,
                precision=precision,
                max_batches=args.eval_batches,
            )

            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "config": old_config,
                    "resumed_from": str(args.resume),
                },
                args.out_dir / f"step_{step:06d}.pt",
            )

        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(json.dumps(record, ensure_ascii=False))

    print("resume smoke done")


if __name__ == "__main__":
    main()
