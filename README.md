# AIResearcher

Kho lưu các dự án học tập về mạng nơ-ron. Mỗi dự án “from scratch” có thư mục riêng dưới `projects/`; bài dịch máy English–Vietnamese là một dự án ứng dụng riêng trong `MachineTraslation/`.

## Lộ trình dự án nhỏ

| Thứ tự | Dự án | Tình trạng hiện tại | Bài tham khảo |
|---|---|---|---|
| 1 | [Neural Network from scratch](projects/01-neural-network-from-scratch/README.md) | Đã chuyển notebook Iris MLP dùng PyTorch vào `experiments/`; bản MNIST/NumPy chưa làm | [Coding Owen](https://codingowen.github.io/projects/neural_network_from_scratch/) |
| 2 | [RNN from scratch](projects/02-rnn-from-scratch/README.md) | Đã có lý thuyết đến mục 2.3 và notebook thử nghiệm ban đầu; code hoàn chỉnh chưa làm | [Coding Owen](https://codingowen.github.io/projects/recurrent_nn_from_scratch/) |
| 3 | [LSTM from scratch](projects/03-lstm-from-scratch/README.md) | Đã tạo không gian dự án; chưa có bản triển khai độc lập | [Coding Owen](https://codingowen.github.io/projects/lstm_from_scratch/) |
| 4 | [Self-Attention from scratch](projects/04-self-attention-from-scratch/README.md) | Đã tạo không gian dự án; chưa có bản triển khai độc lập | [Coding Owen](https://codingowen.github.io/projects/self_attention/) |
| 5 | [Vision Transformer (ViT) from scratch](projects/05-vision-transformer-from-scratch/README.md) | Đã tạo không gian dự án; chưa có bản triển khai độc lập | [Coding Owen](https://codingowen.github.io/projects/vision_transformer/) |

Các link trên là tài liệu học và đối chiếu. Mỗi thư mục có README riêng để bổ sung lý thuyết, công thức, notebook và kết quả khi thực sự hoàn thành; bảng tình trạng phản ánh những gì hiện đã có trong repo.

## Dự án dịch máy

[English-to-Vietnamese Neural Machine Translation](MachineTraslation/README.md) dùng PyTorch để xây dựng baseline Seq2Seq LSTM không attention trước, rồi mới nghiên cứu attention. Dữ liệu IWSLT15, notebook và hình minh họa của dự án này nằm trong `MachineTraslation/`. Tên thư mục hiện tại được giữ để các đường dẫn dữ liệu đã dùng trong notebook vẫn hoạt động.

## Cấu trúc

```text
AIResearcher/
├── projects/
│   ├── 01-neural-network-from-scratch/
│   ├── 02-rnn-from-scratch/
│   ├── 03-lstm-from-scratch/
│   ├── 04-self-attention-from-scratch/
│   └── 05-vision-transformer-from-scratch/
├── MachineTraslation/
│   ├── data/
│   ├── images/
│   ├── notebooks/
│   ├── scripts/
│   └── src/
├── tensor/             # thí nghiệm cá nhân riêng, hiện được gitignore
└── requirements.txt
```

Môi trường Python dùng `requirements.txt` ở gốc repo. Hướng dẫn dữ liệu và notebook NMT nằm trong [README của MachineTraslation](MachineTraslation/README.md).
