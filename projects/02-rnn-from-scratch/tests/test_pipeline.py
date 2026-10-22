"""Focused checks for the manual translation pipeline."""

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data import TranslationDataset, collate_pairs
from src.inference import load_model
from src.model import ModelConfig, Seq2Seq, VanillaRNNCell
from src.train import fit


class PipelineTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.device = torch.device("cpu")

    def test_cell_formula_and_gradients(self):
        cell = VanillaRNNCell(3, 5)
        x, previous = torch.randn(2, 3), torch.randn(2, 5)
        expected = torch.tanh(x @ cell.W_hx.T + previous @ cell.W_hh.T + cell.b_h)
        self.assertTrue(torch.allclose(cell(x, previous), expected))
        cell(x, previous).sum().backward()
        self.assertTrue(all(parameter.grad is not None for parameter in cell.parameters()))

    def test_padding_and_batch_contract(self):
        pairs = [([1, 4, 2], [1, 6, 2]), ([1, 5, 7, 2], [1, 8, 9, 2])]
        source, lengths, target_in, target_out = collate_pairs(
            [TranslationDataset(pairs)[index] for index in range(2)])
        self.assertEqual(target_in.tolist(), [[1, 6, 2], [1, 8, 9]])
        self.assertEqual(target_out.tolist(), [[6, 2, 0], [8, 9, 2]])
        model = Seq2Seq(ModelConfig(12, 13, 4, 5))
        state = model.encoder(source, lengths)
        padded_state = model.encoder(nn.functional.pad(source, (0, 3)), lengths)
        self.assertTrue(torch.allclose(state, padded_state))
        logits = model(source, lengths, target_in)
        self.assertEqual(tuple(logits.shape), (2, 3, 13))
        criterion = nn.CrossEntropyLoss(ignore_index=0, reduction="sum")
        loss = criterion(logits.reshape(-1, 13), target_out.reshape(-1))
        self.assertTrue(torch.isfinite(loss))

    def test_training_checkpoint_and_eos(self):
        pairs = [([1, 4, 2], [1, 6, 2])] * 4
        loader = DataLoader(TranslationDataset(pairs), batch_size=4, collate_fn=collate_pairs)
        model = Seq2Seq(ModelConfig(10, 10, 4, 8))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best.pt"
            best = fit(model, loader, loader, checkpoint, self.device, max_epochs=12,
                       patience=12, learning_rate=0.03)
            self.assertLess(best, 1.0)
            restored = load_model(checkpoint, self.device)
            batch = next(iter(loader))
            self.assertTrue(torch.allclose(model(*batch[:3]), restored(*batch[:3])))
            with torch.no_grad():
                restored.decoder.W_yh.zero_()
                restored.decoder.b_y.fill_(-10)
                restored.decoder.b_y[2] = 10
            self.assertEqual(restored.greedy_decode(batch[0], batch[1]).tolist(), [[2]] * 4)


if __name__ == "__main__":
    unittest.main()
