from collections import Counter

class Vocab:
    def __init__(self, tokenized_sentences, min_freq=2, max_size=30000):
        counter = Counter()
        for tokens in tokenized_sentences:
            counter.update(tokens) # for example : tokens = ["I", "love", "AI"]

        self.word2idx = {
            '<pad>': 0, 
            '<bos>': 1, 
            '<eos>': 2, 
            '<unk>': 3
        }
        self.MOST_COMMON = counter.most_common(1)

        print(f"Most commom {self.MOST_COMMON}")
        for word, freq in counter.most_common(max_size):
            if freq > min_freq:
                self.word2idx[word] = len(self.word2idx)

        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def encode(self, tokens):
        ids = [self.word2idx.get(t, self.word2idx['<unk>']) for t in tokens]
        return [self.word2idx['<bos>']] + ids + [self.word2idx['<eos>']]

    def decode(self, ids):
        return [self.idx2word[i] for i in ids if i not in (0, 1, 2)]

    def __len__(self):
        return len(self.word2idx)
