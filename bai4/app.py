import tkinter as tk
from tkinter import ttk
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Đọc ảnh từ một patch cụ thể
def tai_anh_tu_patch(thu_muc_patch):
    images = []
    for filename in os.listdir(thu_muc_patch):
        img = cv2.imread(os.path.join(thu_muc_patch, filename), cv2.IMREAD_GRAYSCALE)  # Đọc ảnh xám
        if img is not None:
            img_equalized = cv2.equalizeHist(img)  # Cân bằng histogram
            img_resized = cv2.resize(img_equalized, (64, 64))  # Resize ảnh
            images.append(img_resized.flatten())  # Chuyển ảnh thành vector
    return np.array(images)

# Hàm để tải các patch
def tai_patch_anh(patch):
    thu_muc = os.path.join('images', f'patch {patch}')  # Thư mục chứa patch ảnh
    return tai_anh_tu_patch(thu_muc)

# Hàm để chạy mô hình và đưa ra kết quả
def chay_mo_hinh(train_ratio, test_ratio, tree, patch):
    # Xóa các dòng cũ trong bảng
    for row in tree.get_children():
        tree.delete(row)
    
    # Load ảnh từ patch được chọn
    X = tai_patch_anh(patch)
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Tạo nhãn giả để mô hình có thể học
    y = np.zeros(len(X))  # Sử dụng nhãn giả, tất cả là 0 vì chỉ có một lớp
    
    # Chia tập dữ liệu
    X_train, X_test, _, _ = train_test_split(X, y, test_size=test_ratio, random_state=42)
    
    # Khởi tạo mô hình
    svm = SVC(kernel='linear', C=1.0, gamma='auto')
    knn = KNeighborsClassifier(n_neighbors=7)
    
    # Huấn luyện và đánh giá SVM
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    
    # Huấn luyện và đánh giá KNN
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    accuracy_knn = accuracy_score(y_test, y_pred_knn)
    
    # Thêm kết quả vào bảng
    tree.insert("", "end", values=("SVM", round(accuracy_svm, 2)))
    tree.insert("", "end", values=("KNN", round(accuracy_knn, 2)))

# Tạo giao diện với Tkinter
def tao_giao_dien():
    window = tk.Tk()
    window.title("Phân loại y tế - So sánh SVM và K-NN")
    window.geometry("900x600")
    window.config(bg="#f0f0f0")

    # Tinh chỉnh style cho giao diện
    style = ttk.Style()
    style.configure("TButton", font=("Helvetica", 12), padding=10)
    style.configure("Treeview.Heading", font=("Helvetica", 12, "bold"))

    # Tiêu đề
    tieu_de = tk.Label(window, text="Mô hình phân loại y tế - SVM và K-NN", font=("Helvetica", 20, "bold"), bg="#f0f0f0", fg="#333")
    tieu_de.pack(pady=20)

    # Frame chứa các nút tỷ lệ chia train-test
    khung_nut = tk.Frame(window, bg="#f0f0f0")
    khung_nut.pack(pady=10)

    # Bảng kết quả
    cot = ("Mô hình", "Độ chính xác")
    tree = ttk.Treeview(window, columns=cot, show="headings", height=8)
    tree.pack(pady=20)

    # Định dạng các cột
    for col in cot:
        tree.heading(col, text=col)
        tree.column(col, anchor=tk.CENTER, width=140)
    
    # Các nút chọn patch ảnh và tỷ lệ train-test
    def tao_nut(text, train_ratio, test_ratio, patch):
        nut = ttk.Button(khung_nut, text=text, command=lambda: chay_mo_hinh(train_ratio, test_ratio, tree, patch))
        nut.pack(side=tk.LEFT, padx=15)

    # Chọn các patch ảnh và tỷ lệ chia train-test
    tao_nut("Patch 1 - 80-20", 0.8, 0.2, 1)
    tao_nut("Patch 1 - 70-30", 0.7, 0.3, 1)
    tao_nut("Patch 1 - 60-40", 0.6, 0.4, 1)
    tao_nut("Patch 1 - 40-60", 0.4, 0.6, 1)
    
    tao_nut("Patch 2 - 80-20", 0.8, 0.2, 2)
    tao_nut("Patch 2 - 70-30", 0.7, 0.3, 2)
    tao_nut("Patch 2 - 60-40", 0.6, 0.4, 2)
    tao_nut("Patch 2 - 40-60", 0.4, 0.6, 2)
    
    tao_nut("Patch 3 - 80-20", 0.8, 0.2, 3)
    tao_nut("Patch 3 - 70-30", 0.7, 0.3, 3)
    tao_nut("Patch 3 - 60-40", 0.6, 0.4, 3)
    tao_nut("Patch 3 - 40-60", 0.4, 0.6, 3)

    # Khởi chạy giao diện
    window.mainloop()

# Gọi hàm tạo giao diện
tao_giao_dien()
