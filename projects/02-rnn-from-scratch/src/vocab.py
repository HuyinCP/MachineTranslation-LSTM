"""Vocabulary used by the English-Vietnamese preprocessing notebooks.

The class stays at ``src.vocab.Vocab`` so existing vocabulary pickle files
can be loaded from this project's ``data/processed`` directory.
"""

from collections import Counter


class Vocab:
    """Map tokens to IDs while reserving IDs for sequence markers."""

    def __init__(self, tokenized_sentences, min_freq=2, max_size=30000):
        counts = Counter(token for sentence in tokenized_sentences for token in sentence)
        self.word2idx = {
            "<pad>": 0,
            "<bos>": 1,
            "<eos>": 2,
            "<unk>": 3,
        }

        # Preserve the original filtering rule and pickle attribute layout.
        self.MOST_COMMON = counts.most_common(1)
        for token, frequency in counts.most_common(max_size):
            if frequency > min_freq:
                self.word2idx[token] = len(self.word2idx)

        self.idx2word = {index: token for token, index in self.word2idx.items()}

    def encode(self, tokens):
        """Add sequence markers and replace unseen tokens with ``<unk>``."""
        token_ids = [self.word2idx.get(token, self.word2idx["<unk>"]) for token in tokens]
        return [self.word2idx["<bos>"], *token_ids, self.word2idx["<eos>"]]

    def decode(self, ids):
        """Return tokens without padding or sequence markers."""
        return [self.idx2word[index] for index in ids if index not in (0, 1, 2)]

    def __len__(self):
        return len(self.word2idx)
