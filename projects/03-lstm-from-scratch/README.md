# LSTM from scratch

**Tham khảo:** [Building Long Short Term Memory (LSTM) from scratch](https://codingowen.github.io/projects/lstm_from_scratch/).

Thư mục này dành cho dự án LSTM độc lập. Chưa có phần triển khai “from scratch” tại đây.

## Dự kiến xây dựng

1. Định nghĩa $x_t$, $h_{t-1}$, $c_{t-1}$ và kích thước các ma trận theo quy ước trong `AGENTS.md`.
2. Giải thích bốn cổng, trạng thái ứng viên $\tilde{c}_t$, cập nhật $c_t$ và $h_t$.
3. Viết một bước LSTM bằng PyTorch và đối chiếu với `nn.LSTMCell`; PyTorch gộp trọng số của cả bốn cổng trong `weight_ih` và `weight_hh`.
4. Kiểm tra chuỗi nhiều bước và gradient.

Phần NMT hiện có [notebook giải thích LSTM và Seq2Seq](../../MachineTraslation/notebooks/04_model_rnn_basics.ipynb). Notebook đó vẫn thuộc dự án dịch máy; tài liệu và code độc lập sẽ được phát triển ở đây.
