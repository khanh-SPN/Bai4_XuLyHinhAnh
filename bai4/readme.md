
# Phân loại y tế - So sánh mô hình SVM và K-NN

## Giới thiệu
Dự án này thực hiện phân lớp trên bộ dữ liệu y tế bằng cách sử dụng hai thuật toán phân loại chính:
- **SVM (Support Vector Machine)**
- **K-NN (K-Nearest Neighbors)**

Các mô hình này được huấn luyện và kiểm thử với các kịch bản chia tập dữ liệu train-test khác nhau. Kết quả của mỗi mô hình sẽ được so sánh dựa trên các chỉ số:
- **Độ chính xác (Accuracy)**
- **Precision** cho từng lớp
- **Recall** cho từng lớp

## Chức năng
1. **Huấn luyện và kiểm thử mô hình**: Chương trình huấn luyện và kiểm thử các mô hình SVM và K-NN trên các tập dữ liệu chia tỷ lệ train-test khác nhau (80-20, 70-30, 60-40, 40-60).
2. **Hiển thị kết quả**: Kết quả của cả hai mô hình sẽ được hiển thị trong bảng trực quan. Bảng này hiển thị các chỉ số quan trọng như độ chính xác, precision, và recall cho từng lớp phân loại.
3. **So sánh các mô hình**: Giao diện cho phép người dùng dễ dàng so sánh hiệu năng của hai mô hình phân loại qua các kịch bản khác nhau.

## Logic của chương trình
- **Chia tập dữ liệu**: Dữ liệu hình ảnh sẽ được chia thành các tập huấn luyện (train) và kiểm thử (test) theo các tỷ lệ khác nhau: 80-20, 70-30, 60-40, 40-60.
- **Huấn luyện mô hình**:
  - **SVM**: Thuật toán SVM được sử dụng với hàm kernel tuyến tính (linear kernel).
  - **K-NN**: Thuật toán K-NN sử dụng giá trị k=5 và tính toán khoảng cách giữa các điểm dữ liệu.
- **Hiển thị kết quả**: Kết quả của mỗi mô hình sẽ được hiển thị dưới dạng bảng với các chỉ số độ chính xác, precision và recall cho từng lớp.

## Hướng dẫn sử dụng
1. Chạy chương trình và chọn một trong các tỷ lệ chia train-test (80-20, 70-30, 60-40, 40-60).
2. Kết quả của các mô hình sẽ được hiển thị trong bảng tương ứng.

## Hình ảnh
Các hình ảnh dưới đây minh họa kết quả phân loại của các mô hình tương ứng với các tỷ lệ chia khác nhau.

### Tỷ lệ 80-20
![80-20](./demo/80-20.png)

### Tỷ lệ 70-30
![70-30](./demo/70-30.png)

### Tỷ lệ 60-40
![60-40](./demo/60-40.png)

### Tỷ lệ 40-60
![40-60](./demo/40-60.png)
"""

# Write the content to a README.md file
with open("/mnt/data/README.md", "w") as f:
    f.write(readme_content)

"/mnt/data/README.md has been created with the requested content."

