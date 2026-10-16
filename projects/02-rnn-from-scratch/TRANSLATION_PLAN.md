# Dịch Anh–Việt bằng Seq2Seq RNN

**Mục tiêu:** sau phần lý thuyết trong [README](README.md), xây một encoder–decoder dùng **RNN thường** để dịch Anh–Việt. Dự án này dùng lại dữ liệu đã chuẩn bị cho NMT, nhưng có mô hình và notebook huấn luyện riêng. Đây là một bài thực hành khác bài sinh văn bản trong [nguồn lý thuyết](https://codingowen.github.io/projects/recurrent_nn_from_scratch/).

## Dữ liệu dùng chung

| Thành phần hiện có | Vai trò trong dự án RNN |
|---|---|
| [02_preprocess.ipynb](../../MachineTraslation/notebooks/02_preprocess.ipynb) | Đọc cặp câu Anh–Việt, tokenize, xây `en_vocab`/`vi_vocab` và đổi câu thành token ID. Không cần tokenize lại nếu các file đã xử lý vẫn còn. |
| [data/processed](../../MachineTraslation/data/processed) | Chứa `train_ids.pkl`, `dev_ids.pkl`, `test_ids.pkl`, `en_vocab.pkl`, `vi_vocab.pkl`. Mỗi mẫu là cặp `(src_ids, tgt_ids)`. |
| [03_dataset.ipynb](../../MachineTraslation/notebooks/03_dataset.ipynb) | Minh họa `TranslationDataset`, padding, độ dài câu nguồn và cách dịch target một bước thành `tgt_input`/`tgt_output`. Có thể dùng cùng hợp đồng batch cho RNN. |
| [src/vocab.py](../../MachineTraslation/src/vocab.py) | Định nghĩa `Vocab` được lưu trong các file pickle; khi tải vocabulary cần để module này import được. |

`04_model_rnn_basics.ipynb` trong NMT giải thích **LSTM Seq2Seq** và tải lại dữ liệu đã xử lý. Nó không phải phần triển khai RNN thường của dự án này; phần code RNN sẽ được viết riêng.

## Luồng mô hình dự kiến

1. **Encoder RNN** đọc từng token tiếng Anh và trả về trạng thái ẩn cuối. Trạng thái này tóm tắt câu nguồn để khởi tạo decoder.
2. **Decoder RNN** nhận `<bos>`, trạng thái từ encoder và sinh từng token tiếng Việt. Trong huấn luyện, dùng `tgt_input` từ batch để **Teacher Forcing**; loss so với `tgt_output`, bỏ qua `<pad>`.
3. Khi suy luận, decoder dùng token nó vừa dự đoán làm đầu vào bước kế tiếp và dừng ở `<eos>` hoặc giới hạn độ dài.

Chỉ truyền trạng thái cuối từ encoder sang decoder tạo **nút thắt trạng thái cuối**: mọi thông tin câu nguồn phải nằm trong một vector cố định. RNN thường còn dễ gặp gradient biến mất trên câu dài, nên đây là baseline học tập để so sánh với [Seq2Seq LSTM không attention](../../MachineTraslation/README.md). Không thêm attention vào phiên bản đầu tiên.

## Phạm vi triển khai sau phần lý thuyết

- Viết RNN cell, encoder và decoder rõ ràng trong dự án này; không lấy code từ `tensor/`.
- Tải trực tiếp dữ liệu đã xử lý, tránh sao chép các file `.pkl` sang `projects/`.
- Kiểm tra shape và một bước forward trước khi huấn luyện, rồi đánh giá trên dev/test của cùng bộ dữ liệu.
- Giữ notebook preprocess/dataset gốc tại `MachineTraslation/notebooks/` vì pipeline dữ liệu đó còn phục vụ baseline LSTM.

Tài liệu này mô tả quyết định và đường dẫn dùng chung; phần code sẽ được thực hiện sau.
