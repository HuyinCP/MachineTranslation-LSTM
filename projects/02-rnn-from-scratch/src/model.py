"""One-layer vanilla RNN encoder and decoder written from matrix operations."""

from dataclasses import dataclass

import torch
from torch import nn

from .data import BOS_ID, EOS_ID, PAD_ID


@dataclass(frozen=True)
class ModelConfig:
    src_vocab_size: int
    tgt_vocab_size: int
    embedding_dim: int = 128
    hidden_dim: int = 256


class VanillaRNNCell(nn.Module):
    """Apply h_t = tanh(W_hx x_t + W_hh h_(t-1) + b_h).

    Parameters retain the column-vector shapes from the project notation.
    The transposes below adapt them to PyTorch batches of row vectors.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.W_hx = nn.Parameter(torch.empty(hidden_dim, input_dim))
        self.W_hh = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.b_h = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.xavier_uniform_(self.W_hx)
        nn.init.orthogonal_(self.W_hh)

    def preactivation(self, embedding_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """Return z_h for one batch of time-step inputs."""
        return embedding_t @ self.W_hx.T + h_prev @ self.W_hh.T + self.b_h

    def forward(self, embedding_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.preactivation(embedding_t, h_prev))


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_ID)
        self.cell = VanillaRNNCell(embedding_dim, hidden_dim)

    def forward(self, src_padded: torch.Tensor, src_lengths: torch.Tensor) -> torch.Tensor:
        """Return each sentence's state after its last non-padding token."""
        batch_size, steps = src_padded.shape
        if src_lengths.shape != (batch_size,) or (src_lengths < 1).any() or (src_lengths > steps).any():
            raise ValueError("src_lengths must contain one valid length per source sentence")
        embeddings = self.embedding(src_padded)
        h_t = embeddings.new_zeros(batch_size, self.cell.hidden_dim)
        for step in range(steps):
            candidate = self.cell(embeddings[:, step], h_t)
            active = (step < src_lengths).unsqueeze(1)
            h_t = torch.where(active, candidate, h_t)
        return h_t


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_ID)
        self.cell = VanillaRNNCell(embedding_dim, hidden_dim)
        self.W_yh = nn.Parameter(torch.empty(vocab_size, hidden_dim))
        self.b_y = nn.Parameter(torch.zeros(vocab_size))
        nn.init.xavier_uniform_(self.W_yh)

    def step(self, token_ids: torch.Tensor, h_prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Consume one reference or predicted token and return logits and state."""
        embedding_t = self.embedding(token_ids)
        h_t = self.cell(embedding_t, h_prev)
        logits_t = h_t @ self.W_yh.T + self.b_y
        return logits_t, h_t


class Seq2Seq(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config.src_vocab_size, config.embedding_dim, config.hidden_dim)
        self.decoder = Decoder(config.tgt_vocab_size, config.embedding_dim, config.hidden_dim)

    def forward(
        self, src_padded: torch.Tensor, src_lengths: torch.Tensor, tgt_input: torch.Tensor
    ) -> torch.Tensor:
        """Return decoder logits [batch, target_steps, target_vocab_size]."""
        if tgt_input.ndim != 2 or tgt_input.shape[0] != src_padded.shape[0] or tgt_input.shape[1] < 1:
            raise ValueError("tgt_input must have shape [batch, target_steps] with at least one step")
        h_t = self.encoder(src_padded, src_lengths)
        outputs = []
        for step in range(tgt_input.shape[1]):
            logits_t, h_t = self.decoder.step(tgt_input[:, step], h_t)
            outputs.append(logits_t)
        return torch.stack(outputs, dim=1)

    @torch.no_grad()
    def greedy_decode(
        self, src_padded: torch.Tensor, src_lengths: torch.Tensor, max_new_tokens: int = 51
    ) -> torch.Tensor:
        """Generate token IDs after <bos>, stopping once every sentence reaches <eos>."""
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        h_t = self.encoder(src_padded, src_lengths)
        token_ids = torch.full((src_padded.shape[0],), BOS_ID, dtype=torch.long, device=src_padded.device)
        finished = torch.zeros_like(token_ids, dtype=torch.bool)
        predictions = []
        for _ in range(max_new_tokens):
            logits_t, h_t = self.decoder.step(token_ids, h_t)
            logits_t[:, PAD_ID] = -torch.inf
            logits_t[:, BOS_ID] = -torch.inf
            token_ids = logits_t.argmax(dim=-1)
            token_ids = torch.where(finished, EOS_ID, token_ids)
            predictions.append(token_ids)
            finished |= token_ids.eq(EOS_ID)
            if finished.all():
                break
        return torch.stack(predictions, dim=1)
