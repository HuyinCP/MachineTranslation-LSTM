# English-to-Vietnamese Neural Machine Translation

This project builds an English-to-Vietnamese neural machine translation system with PyTorch. It is written as a learning project, so the notebooks explain the main ideas behind the model instead of hiding everything inside a high-level training framework.

The project starts with the original Seq2Seq architecture: an LSTM encoder reads an English sentence and a decoder generates the Vietnamese translation. This first version does not use attention. That limitation is intentional because it makes the information bottleneck of the original Seq2Seq model easier to understand before studying attention-based translation.

## The Translation Problem

The model receives a source sentence in English and predicts a target sentence in Vietnamese. For example:

```text
English:  I am a student.
Vietnamese: Tôi là một sinh viên.
```

Every sentence pair is converted into a sequence of token IDs. The source sequence is given to the encoder. The decoder then predicts one Vietnamese token at a time until it produces the `<eos>` token.

The dataset used by this project is the IWSLT15 English-Vietnamese corpus. The raw files are stored in `MachineTraslation/data/raw/` and contain separate English and Vietnamese files for the training, development, and test splits.

## How the Solution Works

### Preparing the data

The preprocessing pipeline performs the following operations:

1. Read the parallel English and Vietnamese sentences.
2. Tokenize English with a small regex-based tokenizer.
3. Tokenize Vietnamese with `underthesea`.
4. Build one vocabulary for each language using the training split.
5. Remove empty or overly long sentence pairs.
6. Convert tokens into integer IDs and add `<bos>` and `<eos>` markers.
7. Save the vocabularies and processed sentence pairs in `data/processed/`.

The vocabulary reserves four special tokens:

```text
<pad>  0
<bos>  1
<eos>  2
<unk>  3
```

### Batching variable-length sentences

Sentences do not all have the same length, so a batch is padded to the length of its longest sentence. The dataset notebook creates:

- `src_batch`: padded English input for the encoder;
- `src_lengths`: original source lengths before padding;
- `tgt_input`: decoder input beginning with `<bos>`;
- `tgt_output`: training targets ending with `<eos>`.

The decoder input and target are shifted by one position. For a target sequence such as:

```text
<bos> I am a student <eos>
```

the decoder receives:

```text
<bos> I am a student
```

and the expected output is:

```text
I am a student <eos>
```

### The LSTM encoder and decoder

An LSTM keeps two states at each time step: the hidden state `$h_t$` and the cell state `$c_t$`. The cell state provides an additive memory path that helps information and gradients travel across long sequences.

Using the row-vector convention, the four LSTM components are:

$$
\begin{aligned}
i_t &= \sigma(x_t W_{xi} + h_{t-1} W_{hi} + b_i), \\
f_t &= \sigma(x_t W_{xf} + h_{t-1} W_{hf} + b_f), \\
\tilde{c}_t &= \tanh(x_t W_{xc} + h_{t-1} W_{hc} + b_c), \\
o_t &= \sigma(x_t W_{xo} + h_{t-1} W_{ho} + b_o).
\end{aligned}
$$

The cell state and hidden state are updated as follows:

$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
$$

$$
h_t = o_t \odot \tanh(c_t)
$$

![LSTM cell architecture](images/architecture.png)

The encoder reads the English sequence and passes its final hidden state and cell state to the decoder. The decoder uses those states to begin generating the Vietnamese sequence. Because the encoder passes only its final states, the entire source sentence must be compressed into a fixed-size representation. This is the final-state bottleneck of the plain Seq2Seq model, and it becomes especially difficult for long sentences.

### Teacher Forcing

During training, the decoder usually receives the correct previous Vietnamese token as its next input. This is called Teacher Forcing. It gives the decoder a reliable input at every step while it learns the mapping from the encoder state to the target sentence.

At inference time, the correct target sentence is unavailable. The decoder must instead feed its own previous prediction into the next step. This difference between training and inference is an important property of Seq2Seq models.

![Teacher Forcing](images/teacher_forcing.png)

## Notebook Guide

The notebooks follow the data and model flow:

| Notebook | Purpose |
|---|---|
| [`02_preprocess.ipynb`](MachineTraslation/notebooks/02_preprocess.ipynb) | Read the raw corpus, tokenize both languages, build vocabularies, filter sentence pairs, and save processed data. |
| [`03_dataset.ipynb`](MachineTraslation/notebooks/03_dataset.ipynb) | Load processed pairs, define `TranslationDataset`, pad batches, and create decoder inputs and targets. |
| [`04_model_rnn_basics.ipynb`](MachineTraslation/notebooks/04_model_rnn_basics.ipynb) | Explain the LSTM gates and the plain Seq2Seq data setup without attention. |
| [`04_test.ipynb`](MachineTraslation/notebooks/04_test.ipynb) | Compare a manually assembled LSTM step with `nn.LSTMCell` and sequence processing with `nn.LSTM`. |

The manual LSTM implementation is used as a conceptual check. PyTorch combines the four gate parameters internally in `nn.LSTMCell`; `weight_ih` and `weight_hh` each contain parameters for all four gates rather than a single gate.

## Project Structure

```text
D:\AIResearcher\
|-- images\
|-- MachineTraslation\
|   |-- data\
|   |   |-- raw\
|   |   |-- processed\
|   |-- notebooks\
|   |-- scripts\
|   |-- src\
|       |-- vocab.py
|-- tensor\
|-- requirements.txt
|-- README.md
```

The `tensor/` directory contains separate experiments and is not used by the NMT implementation.

## Setup and Usage

Create a virtual environment and install the dependencies:

```powershell
python -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
```

To download the IWSLT15 files, run the download script from Git Bash or WSL:

```bash
bash MachineTraslation/scripts/download_data.sh
```

Then open the notebooks in `MachineTraslation/notebooks/` and run them in order. The preprocessing notebook should be run before the dataset and model notebooks because the later notebooks load the files written to `MachineTraslation/data/processed/`.

The project uses Python, PyTorch, NumPy, `underthesea`, and Jupyter/IPython. PyTorch's automatic differentiation is used for model computation; the LSTM equations are implemented manually for understanding and comparison, not for replacing autograd.
