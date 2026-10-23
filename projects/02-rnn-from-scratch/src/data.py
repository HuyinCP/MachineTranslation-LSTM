"""Load the saved parallel corpus and make batches for Seq2Seq training."""

import pickle
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from .vocab import Vocab

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

SentencePair = tuple[list[int], list[int]]


@dataclass
class Corpus:
    en_vocab: Vocab
    vi_vocab: Vocab
    train_pairs: list[SentencePair]
    dev_pairs: list[SentencePair]
    test_pairs: list[SentencePair]


def _load_pickle(path: Path):
    with path.open("rb") as stream:
        return pickle.load(stream)


def load_corpus(data_dir: Path) -> Corpus:
    """Load project-local files and check the token-ID contract."""
    processed = Path(data_dir) / "processed"
    corpus = Corpus(
        en_vocab=_load_pickle(processed / "en_vocab.pkl"),
        vi_vocab=_load_pickle(processed / "vi_vocab.pkl"),
        train_pairs=_load_pickle(processed / "train_ids.pkl"),
        dev_pairs=_load_pickle(processed / "dev_ids.pkl"),
        test_pairs=_load_pickle(processed / "test_ids.pkl"),
    )
    expected = {"<pad>": PAD_ID, "<bos>": BOS_ID, "<eos>": EOS_ID, "<unk>": UNK_ID}
    for name, vocab in (("English", corpus.en_vocab), ("Vietnamese", corpus.vi_vocab)):
        if any(vocab.word2idx.get(token) != index for token, index in expected.items()):
            raise ValueError(f"{name} vocabulary has unexpected special-token IDs")
    for split, pairs in (("train", corpus.train_pairs), ("dev", corpus.dev_pairs), ("test", corpus.test_pairs)):
        if not pairs:
            raise ValueError(f"{split} split is empty")
        if any(not src or not tgt or src[0] != BOS_ID or src[-1] != EOS_ID
               or tgt[0] != BOS_ID or tgt[-1] != EOS_ID for src, tgt in pairs):
            raise ValueError(f"{split} split contains a malformed sentence pair")
    return corpus


class TranslationDataset(Dataset):
    def __init__(self, pairs: list[SentencePair]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        source, target = self.pairs[index]
        return torch.tensor(source, dtype=torch.long), torch.tensor(target, dtype=torch.long)


def collate_pairs(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, ...]:
    """Return source IDs, true source lengths, and shifted decoder IDs."""
    sources, targets = zip(*batch)
    src_lengths = torch.tensor([len(source) for source in sources], dtype=torch.long)
    src_padded = pad_sequence(sources, batch_first=True, padding_value=PAD_ID)
    tgt_padded = pad_sequence(targets, batch_first=True, padding_value=PAD_ID)
    return src_padded, src_lengths, tgt_padded[:, :-1], tgt_padded[:, 1:]


def make_loaders(corpus: Corpus, batch_size: int = 64) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Shuffle training pairs while keeping development and test order stable."""
    return (
        DataLoader(TranslationDataset(corpus.train_pairs), batch_size=batch_size, shuffle=True, collate_fn=collate_pairs),
        DataLoader(TranslationDataset(corpus.dev_pairs), batch_size=batch_size, shuffle=False, collate_fn=collate_pairs),
        DataLoader(TranslationDataset(corpus.test_pairs), batch_size=batch_size, shuffle=False, collate_fn=collate_pairs),
    )
