# IMDB: RNN, LSTM, SVM và Logistic Regression

Bài tập phân tích cảm xúc nhị phân: `0 = negative`, `1 = positive`.
Code dùng **PyTorch**, **scikit-learn**, NumPy và Matplotlib; không cần torchtext,
Transformers hay tải thêm bộ tokenizer. Tokenizer dùng regex của Python.

## Các file

- `data.py`: tải/đọc IMDB, tách validation, tokenize, tạo vocabulary và DataLoader.
- `train.py`: TF-IDF + Logistic Regression/Linear SVM; Embedding + RNN/LSTM;
  huấn luyện, chọn mô hình và xuất kết quả so sánh.
- `requirements.txt`: thư viện cần cài.

## Cách chạy

Chạy trong thư mục chứa các file, với Python 3.10 trở lên và bản PyTorch hỗ trợ Python đang dùng:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train.py
```

Lần đầu chương trình tải IMDB từ bản sao fast.ai trên S3 (~144 MB) vào `data/`;
nếu lỗi kết nối sẽ tự thử nguồn Stanford (~84 MB). Không cần cài fastai.
File tải dở không được dùng làm archive hoàn chỉnh.
Nếu cả hai nguồn đều lỗi, tải [imdb.tgz](https://s3.amazonaws.com/fast-ai-nlp/imdb.tgz)
bằng trình duyệt, lưu thành `data/aclImdb_v1.tar.gz`, rồi chạy lại.
Nếu tải thủ công, giải nén để có `data/aclImdb/train/pos/*.txt`,
`train/neg/*.txt`, `test/pos/*.txt`, `test/neg/*.txt`, mỗi thư mục 12.500 file.
Nếu archive fast.ai giải nén ra thư mục `imdb`, đổi tên thành `aclImdb`.
Chương trình tự dùng GPU CUDA nếu PyTorch nhận diện được, còn lại dùng CPU.

Chạy thử trên mẫu nhỏ trước (vẫn cần tải bộ dữ liệu đầy đủ):

```powershell
.\.venv\Scripts\python.exe train.py --limit 1000 --epochs 2 --output-dir outputs_quick
```

Chạy đầy đủ hoặc thay đổi độ dài review:

```powershell
.\.venv\Scripts\python.exe train.py --epochs 10 --max-len 256 --batch-size 64
```

## Thiết lập thí nghiệm

- Dữ liệu gốc: 50.000 review có nhãn, cân bằng hai lớp; bỏ qua phần `unsup`.
- Giữ nguyên test gốc 25.000 mẫu. Chia train gốc thành 20.000 train và
  5.000 validation, có phân tầng nhãn, seed mặc định 42.
- Cả bốn mô hình dùng cùng split và tokenizer; giữ từ phủ định như `not`, `don't`.
  TF-IDF và vocabulary chỉ học trên train để tránh rò rỉ dữ liệu.
- Baseline: TF-IDF unigram/bigram, tối đa 50.000 features; chọn `C` trong
  `{0.1, 1, 10}` theo accuracy validation. SVM là `LinearSVC`.
- RNN/LSTM một chiều, một tầng: embedding 128 → hidden 128 → dropout 0.3 →
  linear 2 lớp. Vocabulary tối đa 20.000 token; lấy tối đa 256 token đầu/review.
  Padding được bỏ qua bằng `pack_padded_sequence`; review rỗng dùng `<unk>`.
- Adam, learning rate 0.001, cross-entropy, gradient clipping 1.0.
  Dừng sau hai epoch liên tiếp không cải thiện validation loss và phục hồi
  trọng số tại epoch có validation loss thấp nhất trước khi đánh giá test.
- Test chỉ dùng đánh giá cuối, không chọn `C`, epoch hay ngưỡng dự đoán.
  Seed giúp tái lập; kết quả có thể khác nhẹ giữa thiết bị/phiên bản thư viện.

## Kết quả và phần nhận xét

Kết quả lần chạy đầy đủ trên CPU (20.000 mẫu train, 5.000 validation và
25.000 test):

| Mô hình | Accuracy | F1 | Thời gian train (giây) |
|---|---:|---:|---:|
| Logistic Regression | **0,8989** | **0,8990** | **0,88** |
| SVM | 0,8958 | 0,8957 | 1,95 |
| RNN | 0,7614 | 0,7650 | 132,23 |
| LSTM | 0,8220 | 0,8274 | 824,10 |

Logistic Regression cho kết quả tốt nhất và nhanh nhất; SVM có chất lượng gần
tương đương. LSTM tốt hơn RNN khoảng 6 điểm phần trăm Accuracy nhờ khả năng ghi
nhớ phụ thuộc dài, nhưng vẫn kém các phương pháp TF-IDF khoảng 7–8 điểm và tốn
nhiều thời gian hơn. Nguyên nhân có thể là mạng neural học embedding từ đầu và
chỉ dùng 256 token đầu, trong khi TF-IDF unigram/bigram khai thác trực tiếp các
cụm từ thể hiện cảm xúc.

Loss validation của RNN tăng sau epoch 6 và của LSTM tăng sau epoch 5 trong khi
loss train tiếp tục giảm, cho thấy dấu hiệu quá khớp. Các số liệu chi tiết và biểu
đồ được lưu trong `outputs/results.csv`, `outputs/history.json` và
`outputs/comparison.png`.

## Tài liệu

- [Stanford Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
- [PyTorch RNN](https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html)
- [PyTorch LSTM](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- [scikit-learn TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
