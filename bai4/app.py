from PIL import Image
import tkinter as tk
from tkinter import ttk
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Đọc ảnh từ thư mục cố định 'images'
def tai_anh_tu_thu_muc(status_label):
    images = []
    status_label.config(text="Đang tải dữ liệu...")
    window.update()  # Cập nhật giao diện để hiển thị trạng thái

    # Đường dẫn cố định đến thư mục 'images'
    thu_muc_anh = Path('./images')  # Đường dẫn cố định đến thư mục 'images'

    if not thu_muc_anh.exists():
        status_label.config(text="Thư mục không tồn tại")
        return np.array([])  # Trả về mảng rỗng nếu thư mục không tồn tại

    # Lấy tất cả các tệp .jpg từ thư mục 'images'
    for filename in thu_muc_anh.glob('*.jpg'):
        file_path = str(filename)  # Chuyển đổi sang chuỗi để Pillow xử lý

        # Thử đọc ảnh bằng Pillow
        try:
            with Image.open(file_path) as img:
                img = img.convert('L')  # Chuyển thành ảnh xám
                img_resized = img.resize((64, 64))  # Resize ảnh
                img_array = np.array(img_resized)  # Chuyển đổi thành mảng numpy
                images.append(img_array.flatten())  # Chuyển ảnh thành vector
        except Exception as e:
            print(f"Không thể đọc tệp: {file_path} - Lỗi: {e}")
            continue

    # Kiểm tra nếu không có ảnh hợp lệ
    if len(images) == 0:
        status_label.config(text="Không có ảnh hợp lệ trong thư mục này")
        return np.array([])

    status_label.config(text="Tải dữ liệu xong")
    return np.array(images)

# Hàm để chạy mô hình và đưa ra kết quả
def chay_mo_hinh(train_ratio, test_ratio, tree, status_label):
    # Xóa các dòng cũ trong bảng
    for row in tree.get_children():
        tree.delete(row)

    # Tải ảnh từ thư mục 'images'
    X = tai_anh_tu_thu_muc(status_label)

    if X.size == 0:
        status_label.config(text="Không có dữ liệu để huấn luyện")
        return  # Dừng nếu không có dữ liệu

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # **Tạo nhãn giả cho 2 lớp**
    # Chia dữ liệu thành 2 lớp: 50 ảnh là lớp 0, 50 ảnh là lớp 1
    y = np.zeros(len(X))
    y[len(X) // 2:] = 1  # Gán 50 ảnh đầu là lớp 0, 50 ảnh sau là lớp 1

    # Cập nhật trạng thái "Đang huấn luyện mô hình..."
    status_label.config(text="Đang huấn luyện mô hình...")
    window.update()  # Cập nhật giao diện

    # Chia tập dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, stratify=y, random_state=42)

    # Khởi tạo mô hình
    svm = SVC(kernel='linear', C=1.0, gamma='auto')
    knn = KNeighborsClassifier(n_neighbors=7)

    # Huấn luyện và đánh giá SVM
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    precision_svm = precision_score(y_test, y_pred_svm, average='weighted', zero_division=0)
    recall_svm = recall_score(y_test, y_pred_svm, average='weighted', zero_division=0)
    f1_svm = f1_score(y_test, y_pred_svm, average='weighted', zero_division=0)

    # Huấn luyện và đánh giá KNN
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    accuracy_knn = accuracy_score(y_test, y_pred_knn)
    precision_knn = precision_score(y_test, y_pred_knn, average='weighted', zero_division=0)
    recall_knn = recall_score(y_test, y_pred_knn, average='weighted', zero_division=0)
    f1_knn = f1_score(y_test, y_pred_knn, average='weighted', zero_division=0)

    # Thêm kết quả vào bảng (SVM)
    tree.insert("", "end", values=("SVM", round(accuracy_svm, 2), round(precision_svm, 2), round(recall_svm, 2), round(f1_svm, 2)))

    # Thêm kết quả vào bảng (KNN)
    tree.insert("", "end", values=("KNN", round(accuracy_knn, 2), round(precision_knn, 2), round(recall_knn, 2), round(f1_knn, 2)))

    # Cập nhật trạng thái hoàn tất
    status_label.config(text="Huấn luyện hoàn tất")

# Hàm để hiển thị các nút tỷ lệ chia sau khi chọn patch
def hien_thi_tu_chon_patch():
    # Hiển thị các nút tỷ lệ chia train-test
    tao_nut("80-20", 0.8, 0.2)
    tao_nut("70-30", 0.7, 0.3)
    tao_nut("60-40", 0.6, 0.4)
    tao_nut("40-60", 0.4, 0.6)

# Tạo nút tỷ lệ chia
def tao_nut(text, train_ratio, test_ratio):
    nut = ttk.Button(khung_patch, text=text, command=lambda: chay_mo_hinh(train_ratio, test_ratio, tree, status_label))
    nut.pack(side=tk.LEFT, padx=15)

# Tạo giao diện với Tkinter
def tao_giao_dien():
    global khung_patch, tree, status_label, window
    window = tk.Tk()
    window.title("Phân loại y tế - So sánh SVM và K-NN")
    window.geometry("1200x600")
    window.config(bg="#f0f0f0")

    # Tinh chỉnh style cho giao diện
    style = ttk.Style()
    style.configure("TButton", font=("Helvetica", 12), padding=10)
    style.configure("Treeview.Heading", font=("Helvetica", 12, "bold"))

    # Tiêu đề
    tieu_de = tk.Label(window, text="Mô hình phân loại y tế - SVM và K-NN", font=("Helvetica", 20, "bold"), bg="#f0f0f0", fg="#333")
    tieu_de.pack(pady=20)

    # Khung chứa các nút lựa chọn tỷ lệ chia train-test
    khung_patch = tk.Frame(window, bg="#f0f0f0")
    khung_patch.pack(pady=10)

    # Bảng kết quả
    cot = ("Mô hình", "Độ chính xác", "Precision", "Recall", "F1-Score")
    tree = ttk.Treeview(window, columns=cot, show="headings", height=8)
    tree.pack(pady=20)

    # Định dạng các cột
    for col in cot:
        tree.heading(col, text=col)
        tree.column(col, anchor=tk.CENTER, width=180)

    # Nút chọn tỷ lệ chia
    hien_thi_tu_chon_patch()

    # Nhãn hiển thị trạng thái
    status_label = tk.Label(window, text="", font=("Helvetica", 12), bg="#f0f0f0", fg="blue")
    status_label.pack(pady=10)

    # Khởi chạy giao diện
    window.mainloop()

# Gọi hàm tạo giao diện
tao_giao_dien()
