# Translation data

This directory contains the English-Vietnamese corpus used by this RNN project. It is a project-local copy, so the notebooks can run without reading or writing `MachineTraslation/data`.

## Raw files

- `raw/train.en` and `raw/train.vi`: parallel training sentences.
- `raw/tst2012.en` and `raw/tst2012.vi`: development sentences.
- `raw/tst2013.en` and `raw/tst2013.vi`: test sentences.
- `raw/*.tgz`: original downloaded archives.

## Processed files

- `processed/en_vocab.pkl` and `processed/vi_vocab.pkl`: token vocabularies.
- `processed/train_ids.pkl`, `processed/dev_ids.pkl`, and `processed/test_ids.pkl`: pairs of source and target token-ID sequences.

Run `../experiments/02_preprocess.ipynb` to regenerate the processed files from the raw sentence files. The saved vocabulary objects use the local `src.vocab.Vocab` class.
