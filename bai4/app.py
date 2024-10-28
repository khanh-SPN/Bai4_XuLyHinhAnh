import tkinter as tk
from tkinter import ttk
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Đọc ảnh và chuyển thành ma trận
def tai_anh_tu_thu_muc(thu_muc):
    images = []
    labels = []
    for filename in os.listdir(thu_muc):
        img = cv2.imread(os.path.join(thu_muc, filename), cv2.IMREAD_GRAYSCALE)  # Đọc ảnh xám
        if img is not None:
            img_resized = cv2.resize(img, (64, 64))  # Resize ảnh
            images.append(img_resized.flatten())  # Chuyển ảnh thành vector
            # Gán nhãn (giả sử phân lớp 0 và 1)
            if "00012" <= filename <= "00200":
                labels.append(0)  # Nhóm 1
            else:
                labels.append(1)  # Nhóm 2
    return np.array(images), np.array(labels)

# Đường dẫn tới thư mục chứa ảnh
thu_muc_anh = 'images'  # Cập nhật lại đường dẫn tới thư mục ảnh là 'images'

# Load ảnh và nhãn
X, y = tai_anh_tu_thu_muc(thu_muc_anh)

# Hàm để chạy mô hình và đưa ra kết quả
def chay_mo_hinh(train_ratio, test_ratio, tree):
    # Xóa các dòng cũ trong bảng
    for row in tree.get_children():
        tree.delete(row)
    
    # Chia tập dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42)
    
    # Khởi tạo mô hình
    svm = SVC(kernel='linear')
    knn = KNeighborsClassifier(n_neighbors=5)
    
    # Huấn luyện và đánh giá SVM
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    report_svm = classification_report(y_test, y_pred_svm, output_dict=True)
    
    # Huấn luyện và đánh giá KNN
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    accuracy_knn = accuracy_score(y_test, y_pred_knn)
    report_knn = classification_report(y_test, y_pred_knn, output_dict=True)
    
    # Thêm kết quả vào bảng
    tree.insert("", "end", values=("SVM", round(accuracy_svm, 2), 
                                   round(report_svm['0']['precision'], 2), 
                                   round(report_svm['0']['recall'], 2), 
                                   round(report_svm['1']['precision'], 2), 
                                   round(report_svm['1']['recall'], 2)))
    
    tree.insert("", "end", values=("KNN", round(accuracy_knn, 2), 
                                   round(report_knn['0']['precision'], 2), 
                                   round(report_knn['0']['recall'], 2), 
                                   round(report_knn['1']['precision'], 2), 
                                   round(report_knn['1']['recall'], 2)))

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
    cot = ("Mô hình", "Độ chính xác", "Precision (Lớp 0)", "Recall (Lớp 0)", "Precision (Lớp 1)", "Recall (Lớp 1)")
    tree = ttk.Treeview(window, columns=cot, show="headings", height=8)
    tree.pack(pady=20)

    # Định dạng các cột
    for col in cot:
        tree.heading(col, text=col)
        tree.column(col, anchor=tk.CENTER, width=140)
    
    # Các nút cho tỷ lệ chia train-test
    def tao_nut(text, train_ratio, test_ratio):
        nut = ttk.Button(khung_nut, text=text, command=lambda: chay_mo_hinh(train_ratio, test_ratio, tree))
        nut.pack(side=tk.LEFT, padx=15)

    # Các tỷ lệ chia train-test
    tao_nut("80-20", 0.8, 0.2)
    tao_nut("70-30", 0.7, 0.3)
    tao_nut("60-40", 0.6, 0.4)
    tao_nut("40-60", 0.4, 0.6)

    # Khởi chạy giao diện
    window.mainloop()

# Gọi hàm tạo giao diện
tao_giao_dien()
