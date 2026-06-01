#!/usr/bin/env python3
import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_ROOT = Path("/mnt/kai_kpfs/weilai/train/pt-data-pipeline")


@dataclass
class GPTConfig:
    vocab_size: int
    seq_len: int
    n_layer: int = 4
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = False


class PackedBinDataset(torch.utils.data.Dataset):
    def __init__(self, path: Path, stored_len: int):
        self.path = Path(path)
        self.stored_len = stored_len
        self.arr = np.memmap(self.path, dtype=np.uint32, mode="r")
        assert self.arr.shape[0] % stored_len == 0, (
            self.path,
            self.arr.shape[0],
            stored_len,
        )
        self.num_sequences = self.arr.shape[0] // stored_len

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        start = idx * self.stored_len
        end = start + self.stored_len
        sample = np.asarray(self.arr[start:end], dtype=np.int64)
        x = torch.from_numpy(sample[:-1].copy()).long()
        y = torch.from_numpy(sample[1:].copy()).long()
        return x, y


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = cfg.dropout

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        hidden = 4 * cfg.n_embd
        self.fc = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.proj = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = self.fc(x)
        x = F.gelu(x, approximate="tanh")
        x = self.proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # Tie token embedding and lm_head weights.
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.seq_len

        pos = torch.arange(0, T, device=idx.device, dtype=torch.long)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss


def count_params(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def evaluate(model, loader, device, precision, max_batches):
    model.eval()
    losses = []

    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.autocast(
            device_type="cuda",
            dtype=precision,
            enabled=device.type == "cuda",
        ):
            _, loss = model(x, y)

        losses.append(float(loss.item()))

    model.train()
    return sum(losses) / max(len(losses), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--packed-dir", type=Path, default=PROJECT_ROOT / "data/stage/packed_glm_seq2048")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "models/checkpoints/smoke_tiny_gpt_v0")
    parser.add_argument("--run-dir", type=Path, default=PROJECT_ROOT / "data/reports/runs/smoke_tiny_gpt_v0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=6)
    parser.add_argument("--n-embd", type=int, default=384)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.run_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads((args.packed_dir / "meta.json").read_text())
    seq_len = int(meta["seq_len"])
    stored_len = int(meta["stored_len"])
    vocab_size = int(meta["vocab_size"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    train_ds = PackedBinDataset(args.packed_dir / "train.bin", stored_len=stored_len)
    val_ds = PackedBinDataset(args.packed_dir / "val.bin", stored_len=stored_len)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    cfg = GPTConfig(
        vocab_size=vocab_size,
        seq_len=seq_len,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=0.0,
        bias=False,
    )

    model = TinyGPT(cfg).to(device)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    run_config = {
        "packed_dir": str(args.packed_dir),
        "out_dir": str(args.out_dir),
        "run_dir": str(args.run_dir),
        "device": str(device),
        "precision": str(precision),
        "seq_len": seq_len,
        "stored_len": stored_len,
        "vocab_size": vocab_size,
        "train_sequences": len(train_ds),
        "val_sequences": len(val_ds),
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "global_batch_tokens": args.batch_size * args.grad_accum * seq_len,
        "max_steps": args.max_steps,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "model": cfg.__dict__,
        "params": count_params(model),
    }

    (args.run_dir / "config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    metrics_path = args.run_dir / "metrics.jsonl"

    print("== run config ==")
    print(json.dumps(run_config, indent=2, ensure_ascii=False))

    model.train()
    data_iter = iter(train_loader)
    t0 = time.time()
    last_time = t0

    optimizer.zero_grad(set_to_none=True)

    for step in range(1, args.max_steps + 1):
        step_loss = 0.0

        for micro in range(args.grad_accum):
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
                loss = loss / args.grad_accum

            loss.backward()
            step_loss += float(loss.item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        now = time.time()
        dt = now - last_time
        last_time = now

        tokens_this_step = args.batch_size * args.grad_accum * seq_len
        tokens_per_sec = tokens_this_step / max(dt, 1e-9)

        record = {
            "step": step,
            "train_loss": step_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "tokens_this_step": tokens_this_step,
            "tokens_per_sec": tokens_per_sec,
            "elapsed_sec": now - t0,
        }

        if torch.cuda.is_available():
            record["gpu_mem_allocated_gb"] = torch.cuda.memory_allocated() / 1024**3
            record["gpu_mem_reserved_gb"] = torch.cuda.memory_reserved() / 1024**3

        if step == 1 or step % args.eval_interval == 0 or step == args.max_steps:
            val_loss = evaluate(
                model,
                val_loader,
                device=device,
                precision=precision,
                max_batches=args.eval_batches,
            )
            record["val_loss"] = val_loss

            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "config": run_config,
            }
            torch.save(ckpt, args.out_dir / f"step_{step:06d}.pt")

        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(json.dumps(record, ensure_ascii=False))

    print("done")


if __name__ == "__main__":
    main()
