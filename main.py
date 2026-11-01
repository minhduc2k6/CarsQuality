import numpy as np # xử lý mảng và chuyển vị ma trận
import matplotlib.pyplot as plt # vẽ đồ thị giảm Loss
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report # hàm đo độ chính xác

# Import dữ liệu từ file data_preprocessing.py
from data_preprocessing import x_train, y_train, Y_train, x_test, y_test, folds
# Import mô hình từ file mlp_model.py
from mlp_model import train_mlp, predict

#------Chuẩn bị kích thước ma trận theo quy ước (d, N) và (C, N)-----
X_train = x_train.T                  # chuyển vị từ (1382, 21) sang (21, 1382)
Y_train_oh = Y_train.T               # chuyển vị từ (1382, 4) sang (4, 1382)

X_test = x_test.T                    # chuyển vị từ (346, 21) sang (21, 346)

#------Khai báo cấu hình mạng MLP-----
d = X_train.shape[0]                 # 21 đặc trưng đầu vào
h = 15                               # 15 nơ-ron lớp ẩn
C = 4                                # 4 nhãn đánh giá xe
eta = 0.5                            # tốc độ học (learning rate)

#------Huấn luyện mô hình-----
print('\n--- Bắt đầu huấn luyện mạng MLP ---')
W1, b1, W2, b2, loss_history = train_mlp(
    X_train, Y_train_oh, d, h, C, eta, max_count=5000
)

#------Đánh giá kết quả trên tập Train và Test-----
y_pred_train = predict(X_train, W1, b1, W2, b2)
y_pred_test = predict(X_test, W1, b1, W2, b2)

print('\nAccuracy trên tập Train:', accuracy_score(y_train, y_pred_train))
print('Accuracy trên tập Test:', accuracy_score(y_test, y_pred_test))



# =====================================================================
# CHẠY KIỂM CHỨNG CHÉO (5-FOLD CROSS VALIDATION)
# =====================================================================
print("\n--- BẮT ĐẦU CHẠY 5-FOLD CROSS VALIDATION ---")

# Mảng lưu trữ kết quả accuracy của từng fold
fold_accuracies = []

# Duyệt qua danh sách 5 folds đã tạo từ hàm get_kfold_splits
for i in range(len(folds)):
    print(f"\n--- Đang huấn luyện Fold {i+1} ---")
    
    # 1. Bóc tách dữ liệu của fold thứ i
    x_tr, y_tr, Y_tr, x_val, y_val, Y_val = folds[i]
    
    # 2. Ép kiểu ma trận chuyển vị (d, N) và (C, N) cho vào mạng MLP
    X_tr_matrix = x_tr.T       # (21, N_train_fold)
    Y_tr_matrix = Y_tr.T       # (4, N_train_fold)
    X_val_matrix = x_val.T     # (21, N_val_fold)
    
    # 3. Huấn luyện mạng MLP từ đầu cho fold này
    # Lưu ý: Hàm train_mlp tự động gọi init_weights nên mỗi fold sẽ học lại từ đầu, không bị rò rỉ kiến thức
    W1_fold, b1_fold, W2_fold, b2_fold, _ = train_mlp(
        X=X_tr_matrix, 
        Y=Y_tr_matrix, 
        d=X_tr_matrix.shape[0], 
        h=15, 
        C=4, 
        eta=0.5, 
        max_count=4000  # Giảm số vòng lặp xuống một chút để CV chạy nhanh hơn
    )
    
    # 4. Dự đoán trên tập Validation (tập bị giấu đi) của fold hiện tại
    y_pred_val = predict(X_val_matrix, W1_fold, b1_fold, W2_fold, b2_fold)
    
    # 5. Tính toán độ chính xác và lưu lại
    acc_fold = accuracy_score(y_val, y_pred_val)
    fold_accuracies.append(acc_fold)
    print(f"-> Accuracy Fold {i+1}: {acc_fold:.4f}")

#------Tổng hợp kết quả-----
tong_acc = 0
for acc in fold_accuracies:
    tong_acc = tong_acc + acc
mean_acc = tong_acc / len(fold_accuracies)

print(f"\n=> KẾT LUẬN: Accuracy trung bình trên 5 Folds là: {mean_acc:.4f}")

#------Đánh giá chuyên sâu trên tập Test-----
print('\n======================================================')
print(' BÁO CÁO ĐÁNH GIÁ CHUYÊN SÂU (TEST SET)')
print('======================================================')

# 1. In Ma trận nhầm lẫn (Confusion Matrix)
cm = confusion_matrix(y_test, y_pred_test)
print('\n1. Ma trận nhầm lẫn (Confusion Matrix):')
print('Các cột là Giá trị dự đoán (Predict) | Các dòng là Giá trị thực tế (True)')
print(cm)

# 2. In Báo cáo phân loại (Classification Report)
# Bao gồm Precision, Recall và F1-Score cho từng class cụ thể
target_names = ['unacc (0)', 'acc (1)', 'good (2)', 'vgood (3)']
report = classification_report(y_test, y_pred_test, target_names=target_names)

print('\n2. Chi tiết các chỉ số F1-Score, Precision, Recall:')
print(report)