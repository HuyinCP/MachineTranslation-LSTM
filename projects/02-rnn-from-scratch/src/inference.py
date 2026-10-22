"""Load a trained Seq2Seq model and translate English sentences."""

import argparse
import sys
from pathlib import Path

import torch

from .data import EOS_ID, load_corpus
from .model import ModelConfig, Seq2Seq
from .text import join_vi_tokens, tokenize_en

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_model(checkpoint_path: Path, device: torch.device) -> Seq2Seq:
    """Reconstruct model parameters from a training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = Seq2Seq(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval()


@torch.no_grad()
def translate_sentence(model, sentence, en_vocab, vi_vocab, device, max_source_tokens=50, max_new_tokens=51):
    """Greedily translate one sentence, excluding sequence markers from text."""
    source_tokens = tokenize_en(sentence)[:max_source_tokens]
    source_ids = en_vocab.encode(source_tokens)
    source = torch.tensor([source_ids], dtype=torch.long, device=device)
    lengths = torch.tensor([len(source_ids)], dtype=torch.long, device=device)
    predicted = model.greedy_decode(source, lengths, max_new_tokens=max_new_tokens)[0].tolist()
    if EOS_ID in predicted:
        predicted = predicted[:predicted.index(EOS_ID)]
    return join_vi_tokens(vi_vocab.decode(predicted))


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints" / "best.pt")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--text", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=51)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corpus = load_corpus(args.data_dir)
    model = load_model(args.checkpoint, device)
    print(translate_sentence(model, args.text, corpus.en_vocab, corpus.vi_vocab, device,
                             max_new_tokens=args.max_new_tokens))


if __name__ == "__main__":
    main()
