"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""

import argparse
import json
import math
from collections import Counter
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import Transformer, make_src_mask, make_tgt_mask


# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need"

    Smoothed target distribution:
        y_smooth = (1 - eps) * one_hot(y) + eps / (vocab_size - 1)

    Args:
        vocab_size (int)  : Number of output classes.
        pad_idx    (int)  : Index of <pad> token — receives 0 probability.
        smoothing  (float): Smoothing factor ε (default 0.1).
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : shape [batch * tgt_len, vocab_size]  (raw model output)
            target : shape [batch * tgt_len]              (gold token indices)

        Returns:
            Scalar loss value.
        """
        log_probs = F.log_softmax(logits, dim=-1)

        with torch.no_grad():
            true_dist = torch.full_like(log_probs, self.smoothing / (self.vocab_size - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            true_dist[:, self.pad_idx] = 0.0

            pad_rows = target.eq(self.pad_idx)
            if pad_rows.any():
                true_dist[pad_rows] = 0.0

        non_pad = target.ne(self.pad_idx)
        if non_pad.sum() == 0:
            return logits.new_tensor(0.0)

        loss = -(true_dist[non_pad] * log_probs[non_pad]).sum()
        loss = loss / non_pad.sum()
        return loss


# ══════════════════════════════════════════════════════════════════════
#   TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    """
    Run one epoch of training or evaluation.

    Args:
        data_iter  : DataLoader yielding (src, tgt) batches of token indices.
        model      : Transformer instance.
        loss_fn    : LabelSmoothingLoss (or any nn.Module loss).
        optimizer  : Optimizer (None during eval).
        scheduler  : NoamScheduler instance (None during eval).
        epoch_num  : Current epoch index (for logging).
        is_train   : If True, perform backward pass and scheduler step.
        device     : 'cpu' or 'cuda'.

    Returns:
        avg_loss : Average loss over the epoch (float).

    """
    del epoch_num  # kept for API compatibility / logging hooks

    model.train(mode=is_train)
    total_loss = 0.0
    total_tokens = 0

    src_pad_idx = getattr(model, "src_pad_idx", 1)
    tgt_pad_idx = getattr(loss_fn, "pad_idx", getattr(model, "tgt_pad_idx", 1))

    for src, tgt in data_iter:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_inp = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        src_mask = make_src_mask(src, pad_idx=src_pad_idx)
        tgt_mask = make_tgt_mask(tgt_inp, pad_idx=tgt_pad_idx)

        if is_train and optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        logits = model(src, tgt_inp, src_mask, tgt_mask)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

        if is_train and optimizer is not None:
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        non_pad_tokens = tgt_out.ne(tgt_pad_idx).sum().item()
        if non_pad_tokens > 0:
            total_loss += loss.item() * non_pad_tokens
            total_tokens += non_pad_tokens

    return total_loss / max(total_tokens, 1)


# ══════════════════════════════════════════════════════════════════════
#   GREEDY DECODING
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.

    Args:
        model        : Trained Transformer.
        src          : Source token indices, shape [1, src_len].
        src_mask     : shape [1, 1, 1, src_len].
        max_len      : Maximum number of tokens to generate.
        start_symbol : Vocabulary index of <sos>.
        end_symbol   : Vocabulary index of <eos>.
        device       : 'cpu' or 'cuda'.

    Returns:
        ys : Generated token indices, shape [1, out_len].
             Includes start_symbol; stops at (and includes) end_symbol
             or when max_len is reached.

    """
    model.eval()
    src = src.to(device)
    src_mask = src_mask.to(device)

    with torch.no_grad():
        memory = model.encode(src, src_mask)
        ys = torch.tensor([[start_symbol]], dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys, pad_idx=getattr(model, "tgt_pad_idx", 1))
            out = model.decode(memory, src_mask, ys, tgt_mask)
            next_word = int(torch.argmax(out[:, -1, :], dim=-1).item())
            ys = torch.cat(
                [ys, torch.tensor([[next_word]], dtype=torch.long, device=device)],
                dim=1,
            )
            if next_word == end_symbol:
                break

    return ys


# ══════════════════════════════════════════════════════════════════════
#   BLEU EVALUATION
# ══════════════════════════════════════════════════════════════════════

def _get_token_from_vocab(vocab, idx: int) -> str:
    if hasattr(vocab, "itos"):
        return vocab.itos[idx]
    if hasattr(vocab, "lookup_token"):
        return vocab.lookup_token(idx)
    if isinstance(vocab, dict):
        inv = {v: k for k, v in vocab.items()}
        return inv.get(idx, "<unk>")
    raise ValueError("Unsupported vocab type for index-to-token lookup")


def _get_index_from_vocab(vocab, token: str, default: int) -> int:
    if hasattr(vocab, "stoi"):
        return vocab.stoi.get(token, default)
    if isinstance(vocab, dict):
        return vocab.get(token, default)
    try:
        return vocab[token]
    except Exception:
        return default


def _ngram_counts(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _corpus_bleu(references: list[list[list[str]]], hypotheses: list[list[str]]) -> float:
    weights = [0.25, 0.25, 0.25, 0.25]
    p_n = []

    for n in range(1, 5):
        matches = 0
        total = 0

        for refs, hyp in zip(references, hypotheses):
            hyp_counts = _ngram_counts(hyp, n)
            total += sum(hyp_counts.values())

            max_ref_counts = Counter()
            for ref in refs:
                ref_counts = _ngram_counts(ref, n)
                for gram, c in ref_counts.items():
                    if c > max_ref_counts[gram]:
                        max_ref_counts[gram] = c

            for gram, c in hyp_counts.items():
                matches += min(c, max_ref_counts.get(gram, 0))

        p_n.append(matches / total if total > 0 else 0.0)

    hyp_len = sum(len(h) for h in hypotheses)
    ref_len = sum(len(refs[0]) for refs in references)

    if hyp_len == 0:
        return 0.0

    bp = 1.0 if hyp_len > ref_len else math.exp(1.0 - (ref_len / hyp_len))

    if min(p_n) == 0.0:
        geo_mean = 0.0
    else:
        geo_mean = math.exp(sum(w * math.log(p) for w, p in zip(weights, p_n)))

    return 100.0 * bp * geo_mean


def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.

    Args:
        model           : Trained Transformer (in eval mode).
        test_dataloader : DataLoader over the test split.
                          Each batch yields (src, tgt) token-index tensors.
        tgt_vocab       : Vocabulary object with idx_to_token mapping.
                          Must support  tgt_vocab.itos[idx]  or
                          tgt_vocab.lookup_token(idx).
        device          : 'cpu' or 'cuda'.
        max_len         : Max decode length per sentence.

    Returns:
        bleu_score : Corpus-level BLEU (float, range 0–100).

    """
    model.eval()

    pad_idx = _get_index_from_vocab(tgt_vocab, "<pad>", getattr(model, "tgt_pad_idx", 1))
    sos_idx = _get_index_from_vocab(tgt_vocab, "<sos>", getattr(model, "tgt_sos_idx", 2))
    eos_idx = _get_index_from_vocab(tgt_vocab, "<eos>", getattr(model, "tgt_eos_idx", 3))

    references: list[list[list[str]]] = []
    hypotheses: list[list[str]] = []

    with torch.no_grad():
        for src_batch, tgt_batch in test_dataloader:
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)

            for i in range(src_batch.size(0)):
                src = src_batch[i : i + 1]
                tgt = tgt_batch[i : i + 1]

                src_mask = make_src_mask(src, pad_idx=getattr(model, "src_pad_idx", 1))
                pred = greedy_decode(
                    model,
                    src,
                    src_mask,
                    max_len=max_len,
                    start_symbol=sos_idx,
                    end_symbol=eos_idx,
                    device=device,
                )

                pred_tokens = []
                for idx in pred.squeeze(0).tolist():
                    if idx in (sos_idx, pad_idx):
                        continue
                    if idx == eos_idx:
                        break
                    pred_tokens.append(_get_token_from_vocab(tgt_vocab, idx))

                ref_tokens = []
                for idx in tgt.squeeze(0).tolist():
                    if idx in (sos_idx, pad_idx):
                        continue
                    if idx == eos_idx:
                        break
                    ref_tokens.append(_get_token_from_vocab(tgt_vocab, idx))

                references.append([ref_tokens])
                hypotheses.append(pred_tokens)

    return _corpus_bleu(references, hypotheses)


# ══════════════════════════════════════════════════════════════════════
# ❺  CHECKPOINT UTILITIES  (autograder loads your model from disk)
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """
    Save model + optimiser + scheduler state to disk.

    The autograder will call load_checkpoint to restore your model.
    Do NOT change the keys in the saved dict.

    Args:
        model     : Transformer instance.
        optimizer : Optimizer instance.
        scheduler : NoamScheduler instance.
        epoch     : Current epoch number.
        path      : File path to save to (default 'checkpoint.pt').

    Saves a dict with keys:
        'epoch', 'model_state_dict', 'optimizer_state_dict',
        'scheduler_state_dict', 'model_config'

    model_config must contain all kwargs needed to reconstruct
    Transformer(**model_config), e.g.:
        {'src_vocab_size': ..., 'tgt_vocab_size': ...,
         'd_model': ..., 'N': ..., 'num_heads': ...,
         'd_ff': ..., 'dropout': ...}
    """
    model_config = getattr(model, "model_config", {
        "src_vocab_size": model.src_vocab_size,
        "tgt_vocab_size": model.tgt_vocab_size,
        "d_model": model.d_model,
        "N": model.N,
        "num_heads": model.num_heads,
        "d_ff": model.d_ff,
        "dropout": model.dropout_rate,
    })

    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "model_config": model_config,
    }

    torch.save(ckpt, path)


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model (and optionally optimizer/scheduler) state from disk.

    Args:
        path      : Path to checkpoint file saved by save_checkpoint.
        model     : Uninitialised Transformer with matching architecture.
        optimizer : Optimizer to restore (pass None to skip).
        scheduler : Scheduler to restore (pass None to skip).

    Returns:
        epoch : The epoch at which the checkpoint was saved (int).

    """
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return int(checkpoint["epoch"])


# ══════════════════════════════════════════════════════════════════════
#   EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment() -> None:
    """
    Backward-compatible default entrypoint.
    Runs training with default configuration and saves best checkpoint.
    """
    run_training_with_config()


def run_training_with_config(
    config_overrides: Optional[dict] = None,
    run_name: str = "baseline",
    mode: str = "train",
    use_wandb: bool = False,
    checkpoint_path: str = "checkpoint.pt",
    vocab_path: str = "vocab.pt",
    save_best: bool = True,
    patience: int = 0,
) -> dict:
    """
    Configurable training entrypoint used by CLI and notebook-style runs.

    Args:
        config_overrides: Dict of config values to override defaults.
        run_name        : Experiment name.
        mode            : Currently supports "train".
        use_wandb       : If True, logs metrics to W&B.
        checkpoint_path : Output checkpoint path.
        vocab_path      : Output vocab artifact path.
        save_best       : If True, keep best val-loss checkpoint at checkpoint_path.
        patience        : Early-stop patience in epochs (0 disables early stopping).

    Returns:
        dict containing best epoch/loss and test BLEU.
    """
    if mode != "train":
        raise ValueError(f"Unsupported mode: {mode}. Only 'train' is implemented.")

    from dataset import Multi30kDataset
    from lr_scheduler import NoamScheduler

    wandb = None
    if use_wandb:
        try:
            import wandb as _wandb
            wandb = _wandb
        except Exception:
            wandb = None

    config = {
        "batch_size": 64,
        "num_epochs": 20,
        "d_model": 512,
        "N": 6,
        "num_heads": 8,
        "d_ff": 2048,
        "dropout": 0.1,
        "warmup_steps": 4000,
        "learning_rate": 1.0,
        "label_smoothing": 0.1,
        "max_len": 100,
        "num_workers": 2,
    }
    if config_overrides:
        config.update(config_overrides)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if wandb is not None:
        wandb.init(project="da6401-a3", name=run_name, config=config)

    train_dataset = Multi30kDataset(split="train")
    val_dataset = Multi30kDataset(
        split="validation",
        src_vocab=train_dataset.src_vocab,
        tgt_vocab=train_dataset.tgt_vocab,
    )
    test_dataset = Multi30kDataset(
        split="test",
        src_vocab=train_dataset.src_vocab,
        tgt_vocab=train_dataset.tgt_vocab,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=config["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
        num_workers=config["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=test_dataset.collate_fn,
        num_workers=config["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    model = Transformer(
        src_vocab_size=len(train_dataset.src_vocab),
        tgt_vocab_size=len(train_dataset.tgt_vocab),
        d_model=config["d_model"],
        N=config["N"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
    ).to(device)

    # Attach indices/tokenizers/vocab for infer-mode compatibility.
    model.src_pad_idx = train_dataset.src_vocab["<pad>"]
    model.tgt_pad_idx = train_dataset.tgt_vocab["<pad>"]
    model.src_sos_idx = train_dataset.src_vocab["<sos>"]
    model.src_eos_idx = train_dataset.src_vocab["<eos>"]
    model.tgt_sos_idx = train_dataset.tgt_vocab["<sos>"]
    model.tgt_eos_idx = train_dataset.tgt_vocab["<eos>"]
    model.src_vocab = train_dataset.src_vocab
    model.tgt_vocab = train_dataset.tgt_vocab
    model.src_tokenizer = train_dataset.de_tokenizer
    model.tgt_tokenizer = train_dataset.en_tokenizer

    # Save vocab artifact required for Transformer() infer-mode setup.
    torch.save(
        {
            "src_itos": train_dataset.src_vocab.itos,
            "tgt_itos": train_dataset.tgt_vocab.itos,
        },
        vocab_path,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    scheduler = NoamScheduler(
        optimizer,
        d_model=config["d_model"],
        warmup_steps=config["warmup_steps"],
    )
    loss_fn = LabelSmoothingLoss(
        vocab_size=len(train_dataset.tgt_vocab),
        pad_idx=train_dataset.tgt_vocab["<pad>"],
        smoothing=config["label_smoothing"],
    )

    best_val_loss = float("inf")
    best_epoch = -1
    no_improve = 0

    for epoch in range(config["num_epochs"]):
        train_loss = run_epoch(
            train_loader,
            model,
            loss_fn,
            optimizer,
            scheduler,
            epoch_num=epoch,
            is_train=True,
            device=device,
        )

        val_loss = run_epoch(
            val_loader,
            model,
            loss_fn,
            optimizer=None,
            scheduler=None,
            epoch_num=epoch,
            is_train=False,
            device=device,
        )

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if save_best:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                no_improve = 0
                save_checkpoint(model, optimizer, scheduler, epoch, path=checkpoint_path)
                print(f"  -> New best checkpoint saved to {checkpoint_path}")
            else:
                no_improve += 1
        else:
            save_checkpoint(model, optimizer, scheduler, epoch, path=checkpoint_path)

        if wandb is not None:
            wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if patience > 0 and no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
            break

    if save_best and best_epoch >= 0:
        _ = load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None)
    else:
        best_epoch = config["num_epochs"] - 1
        best_val_loss = val_loss

    bleu = evaluate_bleu(
        model,
        test_loader,
        train_dataset.tgt_vocab,
        device=device,
        max_len=config["max_len"],
    )

    print(f"Best epoch: {best_epoch}, best val loss: {best_val_loss:.4f}")
    print(f"Test BLEU (best checkpoint): {bleu:.2f}")
    print(f"Saved best model: {checkpoint_path}")
    print(f"Saved vocab: {vocab_path}")

    if wandb is not None:
        wandb.log({"test_bleu": bleu, "best_epoch": best_epoch, "best_val_loss": best_val_loss})
        wandb.finish()

    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_bleu": bleu,
        "checkpoint_path": checkpoint_path,
        "vocab_path": vocab_path,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Transformer for DA6401 Assignment 3")
    parser.add_argument("--mode", type=str, default="train", choices=["train"], help="Execution mode")
    parser.add_argument("--run_name", type=str, default="baseline", help="Experiment name")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--N", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=4000)
    parser.add_argument("--learning_rate", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--max_len", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--patience", type=int, default=6, help="Early-stop patience; 0 disables")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint.pt")
    parser.add_argument("--vocab_path", type=str, default="vocab.pt")
    parser.add_argument("--metrics_path", type=str, default="training_metrics.json")

    parser.add_argument("--save_best", dest="save_best", action="store_true", default=True)
    parser.add_argument("--save_last", dest="save_best", action="store_false", help="Save latest epoch instead of best")

    parser.add_argument("--wandb", action="store_true", default=False, help="Enable Weights & Biases logging")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    config_overrides = {
        "batch_size": args.batch_size,
        "num_epochs": args.epochs,
        "d_model": args.d_model,
        "N": args.N,
        "num_heads": args.num_heads,
        "d_ff": args.d_ff,
        "dropout": args.dropout,
        "warmup_steps": args.warmup_steps,
        "learning_rate": args.learning_rate,
        "label_smoothing": args.label_smoothing,
        "max_len": args.max_len,
        "num_workers": args.num_workers,
    }

    results = run_training_with_config(
        config_overrides=config_overrides,
        run_name=args.run_name,
        mode=args.mode,
        use_wandb=args.wandb,
        checkpoint_path=args.checkpoint_path,
        vocab_path=args.vocab_path,
        save_best=args.save_best,
        patience=args.patience,
    )

    # Persist run summary for easy reporting/submission tracking.
    payload = {
        "run_name": args.run_name,
        "mode": args.mode,
        "config": config_overrides,
        **results,
    }
    with open(args.metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved metrics: {args.metrics_path}")


if __name__ == "__main__":
    main()
