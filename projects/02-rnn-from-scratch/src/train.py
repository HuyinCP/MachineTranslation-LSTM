"""Train the manual Seq2Seq RNN with full Teacher Forcing."""

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from .data import PAD_ID, load_corpus, make_loaders
from .model import ModelConfig, Seq2Seq

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_epoch(
    model: Seq2Seq,
    loader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip: float = 1.0,
    limit_batches: int | None = None,
) -> float:
    """Return cross-entropy averaged over non-padding target tokens."""
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_tokens = 0
    for batch_index, batch in enumerate(loader, start=1):
        src_padded, src_lengths, tgt_input, tgt_output = (item.to(device) for item in batch)
        valid_tokens = tgt_output.ne(PAD_ID).sum()
        with torch.set_grad_enabled(is_training):
            logits = model(src_padded, src_lengths, tgt_input)
            loss_sum = criterion(logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1))
            if is_training:
                optimizer.zero_grad(set_to_none=True)
                (loss_sum / valid_tokens).backward()
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
        total_loss += loss_sum.detach().item()
        total_tokens += valid_tokens.item()
        if is_training and batch_index % 100 == 0:
            print(f"  batch {batch_index}/{len(loader)}: token loss {total_loss / total_tokens:.4f}", flush=True)
        if limit_batches is not None and batch_index >= limit_batches:
            break
    return total_loss / total_tokens


def fit(
    model: Seq2Seq,
    train_loader,
    dev_loader,
    checkpoint_path: Path,
    device: torch.device,
    max_epochs: int = 10,
    patience: int = 3,
    learning_rate: float = 1e-3,
    gradient_clip: float = 1.0,
    limit_train_batches: int | None = None,
    limit_dev_batches: int | None = None,
) -> float:
    """Save the model with the lowest development token loss."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, reduction="sum")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_dev_loss = float("inf")
    epochs_without_improvement = 0
    for epoch in range(1, max_epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer,
                               gradient_clip, limit_train_batches)
        dev_loss = run_epoch(model, dev_loader, criterion, device,
                             limit_batches=limit_dev_batches)
        print(f"epoch {epoch}: train loss {train_loss:.4f}, dev loss {dev_loss:.4f}", flush=True)
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            epochs_without_improvement = 0
            torch.save({
                "model_config": asdict(model.config),
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "dev_loss": dev_loss,
            }, checkpoint_path)
            print(f"  saved best checkpoint: {checkpoint_path}", flush=True)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("  early stopping: development loss did not improve", flush=True)
                break
    return best_dev_loss


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints" / "best.pt")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train-batches", type=int, default=None,
                        help="Small educational smoke run; omit for the full training split.")
    parser.add_argument("--limit-dev-batches", type=int, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corpus = load_corpus(args.data_dir)
    train_loader, dev_loader, _ = make_loaders(corpus, args.batch_size)
    config = ModelConfig(len(corpus.en_vocab), len(corpus.vi_vocab),
                         args.embedding_dim, args.hidden_dim)
    model = Seq2Seq(config)
    print(f"device={device}, train_batches={len(train_loader)}, dev_batches={len(dev_loader)}")
    fit(model, train_loader, dev_loader, args.checkpoint, device, args.max_epochs,
        args.patience, args.learning_rate, args.gradient_clip,
        args.limit_train_batches, args.limit_dev_batches)


if __name__ == "__main__":
    main()
