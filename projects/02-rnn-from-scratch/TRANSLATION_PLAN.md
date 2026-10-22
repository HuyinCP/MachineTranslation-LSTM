# English-Vietnamese Translation with a Vanilla RNN

This project will build a small encoder-decoder translation model with vanilla RNN cells. Its preprocessing code and data live inside this project, so the experiments do not import from `MachineTraslation`.

## Project layout

| Path | Purpose |
|---|---|
| `src/vocab.py`, `src/text.py` | Vocabulary and shared tokenization. |
| `src/data.py` | Checked corpus loading and the four-tensor batch contract. |
| `src/model.py` | Manual RNN cell, encoder, decoder, and greedy decoding. |
| `src/train.py` | Teacher-forced training, dev loss, gradient clipping, and best checkpoint. |
| `src/inference.py`, `src/evaluate.py` | Sentence translation and tokenized SacreBLEU. |
| `data/raw/` | English and Vietnamese sentence files, plus the original archives. |
| `data/processed/` | Pickled vocabularies and token-ID sentence pairs. |
| `experiments/02_preprocess.ipynb` | Read, tokenize, filter, and numericalize the raw sentence pairs. |
| `experiments/03_dataset.ipynb` | Load the processed data and create padded batches. |

The data began as a copy of the existing English-Vietnamese corpus. Future preprocessing runs in this project write only to this project's `data/processed/` directory. The original `MachineTraslation` data remains available for its LSTM baseline.

The existing vocabulary pickle files refer to the Python class `src.vocab.Vocab`. This project provides that same import path and preserves the vocabulary attributes needed to load those files. Start with a fresh notebook kernel when switching between projects so Python does not reuse an already imported `src` package from another project.

## Notebook order

1. Run `02_preprocess.ipynb` when raw data changes or the processed files need to be rebuilt.
2. Run `03_dataset.ipynb` to inspect token IDs, padding, source lengths, and shifted decoder inputs and targets.
3. Follow `04_model.ipynb`, `05_train.ipynb`, and `06_evaluate.ipynb` for formulas, shapes, training, and evaluation.

Run the notebooks with `experiments/` as the working directory. Their short setup cell sets `PROJECT_ROOT` to its parent for imports and data paths.

## Implemented model

The encoder reads English tokens and passes its final hidden state to the decoder. During training, the decoder receives the previous reference token as input (Teacher Forcing) and predicts the next Vietnamese token. During inference, it uses its own previous prediction and stops at `<eos>` or a maximum length.

Passing only the final encoder state creates an information bottleneck, especially for long sentences. This first model remains a plain RNN baseline without attention and does not use code from `tensor/`.

With `projects/02-rnn-from-scratch` as the working directory, run `D:/AIResearcher/venv/Scripts/python.exe -m src.train`. The defaults are embedding size 128, hidden size 256, batch size 64, Adam learning rate 0.001, gradient clipping 1.0, at most 10 epochs, and patience 3. CUDA is selected when available. A complete run writes `checkpoints/best.pt` according to dev loss.

Then run `python -m src.inference --text "How are you?"` and `python -m src.evaluate --split dev`. Only after selecting the checkpoint using dev loss should you run `python -m src.evaluate --split test`. SacreBLEU uses `BLEU(tokenize="none")` because both predictions and saved references use the same Vietnamese token segmentation. This is an internal comparison, not a benchmark against a different preprocessing pipeline.
