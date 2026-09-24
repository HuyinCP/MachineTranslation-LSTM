# NMT English → Vietnamese (RNN + Attention, from scratch)

Dự án học tập: tự cài đặt lại một mô hình dịch máy nơ-ron (NMT) English → Vietnamese
bằng PyTorch, đi theo đúng 2 giai đoạn để hiểu rõ cơ chế trước khi thêm tối ưu:

1. **Giai đoạn 1 — Seq2Seq LSTM thuần, KHÔNG Attention** *(đang làm)*: Encoder nén
   toàn bộ câu nguồn thành một context vector duy nhất (hidden/cell state cuối
   cùng), Decoder giải mã từ đó. Mục tiêu là tự trải nghiệm giới hạn
   "nút thắt cổ chai" (bottleneck) của kiến trúc Seq2Seq gốc.
2. **Giai đoạn 2 — Thêm Luong Attention** *(chưa làm)*: dựa trên
   Luong, Pham, Manning (2015), *"Effective Approaches to Attention-based
   Neural Machine Translation"* — Encoder trả về toàn bộ hidden states, Decoder
   tra cứu qua attention (dot / general / concat) + input feeding, rồi so BLEU
   với baseline để thấy rõ Attention cải thiện điều gì.

Ưu tiên code rõ ràng, có comment giải thích *tại sao*, không tối ưu hiệu năng.
Khi học một cơ chế mới (LSTM cell, Attention...), luôn code tay đối chiếu với
module PyTorch tương ứng (`nn.LSTMCell`) trước khi dùng thẳng module có sẵn
(`nn.LSTM`) cho phần còn lại.

## Dataset

[IWSLT15 English-Vietnamese](https://nlp.stanford.edu/projects/nmt/) (Stanford NLP Group):

| Split | File | Số câu |
|---|---|---|
| train | `train.en` / `train.vi` | 133,317 |
| dev | `tst2012.en` / `tst2012.vi` | 1,553 |
| test | `tst2013.en` / `tst2013.vi` | 1,268 |

Tải bằng `MachineTraslation/scripts/download_data.sh` (dùng `curl -L`, mirror
GitHub `stefan-it/nmt-en-vi`), giải nén vào `MachineTraslation/data/raw/`.

## Cơ chế LSTM (nền tảng cho cả 2 giai đoạn)

RNN thường bị *vanishing gradient* trên câu dài — gradient co dần về 0 qua
nhiều bước, khiến model "quên" thông tin đầu câu. LSTM giải quyết bằng cách
thêm một đường truyền riêng gọi là **cell state** $c_t$, cập nhật qua phép
**cộng** thay vì nhân liên tiếp — giúp gradient ổn định hơn.

Tại mỗi bước thời gian, với input hiện tại $x_t$, hidden state trước
$h_{t-1}$, cell state trước $c_{t-1}$:

$$
\begin{aligned}
i_t &= \sigma(x_t W_{xi} + h_{t-1} W_{hi} + b_i) && \text{Input gate — thêm bao nhiêu thông tin mới} \\
f_t &= \sigma(x_t W_{xf} + h_{t-1} W_{hf} + b_f) && \text{Forget gate — giữ lại bao nhiêu ký ức cũ} \\
\tilde{c}_t &= \tanh(x_t W_{xc} + h_{t-1} W_{hc} + b_c) && \text{Candidate — nội dung mới có thể ghi vào} \\
o_t &= \sigma(x_t W_{xo} + h_{t-1} W_{ho} + b_o) && \text{Output gate — lộ ra bao nhiêu cho hidden state}
\end{aligned}
$$

Cell state và hidden state được cập nhật:

$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
$$

$$
h_t = o_t \odot \tanh(c_t)
$$

Phép **cộng** ở công thức $c_t$ chính là "memory highway" — đường truyền trực
tiếp giúp gradient không bị triệt tiêu qua nhiều bước như RNN thường.

![LSTM cell architecture](images/architecture.png)

*Node vàng = nhân theo phần tử ($\odot$), node xanh lá = cộng theo phần tử.
Sơ đồ khớp trực tiếp với 4 công thức ở trên: `Forget Gate` → $f_t$,
`Input Gate` → $i_t$ và $\tilde{c}_t$, `Output Gate` → $o_t$, đường ngang trên
cùng (`Cell State`) chính là đường cộng $c_{t-1} \to c_t$.*

Chi tiết từng bước (forget → input/candidate → update cell state → output)
được giải thích trong `MachineTraslation/notebooks/04_model_rnn_basics.ipynb`,
cùng phần code tay `nn.LSTMCell` để đối chiếu công thức với implementation
thật của PyTorch.

## Teacher Forcing (dùng khi train Decoder)

![Teacher Forcing](images/teacher_forcing.png)

Khi train, thay vì cho Decoder tự ăn lại token nó vừa đoán ở bước trước
(dễ tích lũy sai số nếu đoán sai sớm), ta "ép" input của bước $t$ là token
**đúng** $y^{(t-1)}$ từ label — đây là lý do gọi là *teacher forcing*. Encoder
chạy hết câu nguồn, trạng thái cuối (`Encoded State`) được dùng để khởi tạo
Decoder; Decoder sau đó sinh từng $\hat{y}^{(t)}$ một, luôn nhận từ **thật**
làm input cho bước kế tiếp trong lúc train (lúc inference thì không còn label
nên phải tự ăn lại $\hat{y}^{(t)}$ — đây cũng là một khác biệt train/inference
cần lưu ý).

## Cấu trúc thư mục

```
D:\AIResearcher\
├── images\                        # hình minh họa dùng chung cho notebook (README này)
├── MachineTraslation\
│   ├── data\
│   │   ├── raw\                   # 6 file IWSLT15 en-vi đã tải + giải nén
│   │   └── processed\             # vocab (.pkl) + dữ liệu đã numerical hóa
│   ├── notebooks\
│   │   ├── 02_preprocess.ipynb    # ✅ tokenize (EN: regex, VI: underthesea), xây Vocab, lưu pickle
│   │   ├── 03_dataset.ipynb       # ✅ Dataset + DataLoader, padding, tách tgt_input/tgt_output
│   │   └── 04_model_rnn_basics.ipynb  # ⬜ đang làm — LSTM cell (lý thuyết + code tay), Encoder/Decoder KHÔNG Attention
│   ├── scripts\
│   │   └── download_data.sh       # tải + giải nén dữ liệu
│   └── src\
│       └── vocab.py               # class Vocab dùng chung giữa các notebook (word2idx/idx2word, encode/decode)
├── tensor\                         # notebook học riêng self-attention/transformer — KHÔNG thuộc project này
├── venv\                           # virtualenv Python
└── requirements.txt
```

Các bước tiếp theo dự kiến: `05_train_baseline` (train + đánh giá Giai đoạn 1)
→ `06_model_attention` (thêm Luong Attention) → `07_train_attention` →
`08_evaluate` (BLEU) → `09_translate` (dịch thử câu mới).

## Setup & chạy

```bash
# tạo venv, cài dependency
python -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt

# tải dữ liệu IWSLT15 en-vi
bash MachineTraslation/scripts/download_data.sh
```

**GPU (tuỳ chọn)**: `pip install torch` mặc định cài bản CPU-only. Nếu có GPU
NVIDIA, cài đúng bản CUDA khớp driver, ví dụ (driver hỗ trợ CUDA 12.6):
```bash
venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu126
```

Chạy các notebook theo đúng thứ tự số trong `MachineTraslation/notebooks/`
bằng Jupyter/VSCode.

## Tech stack

Python + PyTorch (`nn.LSTM`, `nn.LSTMCell`, `nn.Embedding`, autograd để train
thật — không viết tay backprop). Tokenize tiếng Việt bằng `underthesea`. Toàn
bộ pipeline viết dưới dạng Jupyter Notebook.
