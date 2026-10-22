"""Evaluate decoded translations with consistent tokenized SacreBLEU."""

import argparse
import sys
from pathlib import Path

import torch
from sacrebleu.metrics import BLEU

from .data import EOS_ID, TranslationDataset, collate_pairs, load_corpus
from .inference import load_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def evaluate_bleu(model, pairs, vi_vocab, device, batch_size=64, max_new_tokens=51):
    """Score hypothesis and reference strings using the same saved tokenization."""
    from torch.utils.data import DataLoader

    loader = DataLoader(TranslationDataset(pairs), batch_size=batch_size,
                        shuffle=False, collate_fn=collate_pairs)
    hypotheses, references = [], []
    model.eval()
    for source, lengths, _, _ in loader:
        predictions = model.greedy_decode(source.to(device), lengths.to(device),
                                          max_new_tokens=max_new_tokens).cpu().tolist()
        for predicted, (_, target) in zip(predictions, pairs[len(hypotheses):]):
            if EOS_ID in predicted:
                predicted = predicted[:predicted.index(EOS_ID)]
            hypotheses.append(" ".join(vi_vocab.decode(predicted)))
            references.append(" ".join(vi_vocab.decode(target)))
    bleu = BLEU(tokenize="none")
    return bleu.corpus_score(hypotheses, [references]), bleu.get_signature(), list(zip(hypotheses, references))


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints" / "best.pt")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=51)
    parser.add_argument("--limit-examples", type=int)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corpus = load_corpus(args.data_dir)
    pairs = corpus.dev_pairs if args.split == "dev" else corpus.test_pairs
    if args.limit_examples is not None:
        pairs = pairs[:args.limit_examples]
    model = load_model(args.checkpoint, device)
    score, signature, examples = evaluate_bleu(model, pairs, corpus.vi_vocab, device,
                                                args.batch_size, args.max_new_tokens)
    print(f"{args.split} BLEU: {score.score:.2f} ({signature})")
    for hypothesis, reference in examples[:3]:
        print(f"hypothesis: {hypothesis}\nreference:  {reference}")


if __name__ == "__main__":
    main()
