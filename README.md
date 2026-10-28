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

Sau khi hoàn tất, `outputs/` chứa:

- `results.csv`: Accuracy, Precision, Recall, F1, validation accuracy và thời gian.
  Precision/Recall/F1 lấy lớp positive làm lớp dương.
- `comparison.png`: biểu đồ Accuracy và F1 của bốn mô hình.
- `reports.json`: báo cáo từng lớp và confusion matrix (hàng = nhãn thật,
  cột = nhãn dự đoán; thứ tự negative, positive), kèm `C` đã chọn của baseline.
- `history.json`: train loss, validation loss và validation accuracy theo epoch.
- `config.json`: tham số chạy, thiết bị và kích thước các split thực tế.

`train_seconds` gồm huấn luyện và đánh giá validation, bao gồm cả ba lần thử `C`
ở baseline; không tính tải dữ liệu, tạo đặc trưng hay đánh giá test.
Đây là thời gian của quy trình chọn mô hình trên máy chạy, không phải phép đo
tốc độ từng epoch. Mỗi lần chạy ghi đè kết quả trong thư mục output đã chọn.
Code không lưu trọng số sau khi kết thúc.

Trong báo cáo, lấy số đo thực tế từ `results.csv`, chèn `comparison.png`, rồi:

1. So sánh Accuracy/F1 và thời gian của bốn mô hình trên **lần chạy đầy đủ**.
2. Dùng loss validation để nhận xét quá khớp, dùng confusion matrix để xem lỗi hai lớp.
3. Giải thích cơ chế cổng của LSTM so với RNN thường; đối chiếu với kết quả quan sát.
4. Thảo luận TF-IDF n-gram, giới hạn 256 token và embedding học từ đầu.
   LSTM không được bảo đảm thắng baseline; kết luận phải dựa trên số đo.

Chưa có số đo IMDB điền sẵn. `--limit` chỉ dùng kiểm tra luồng chạy;
không dùng kết quả mẫu nhỏ thay cho kết quả của 50.000 review.

## Tài liệu

- [Stanford Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
- [PyTorch RNN](https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html)
- [PyTorch LSTM](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- [scikit-learn TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
