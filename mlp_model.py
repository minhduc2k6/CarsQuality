import numpy as np # tính toán đại số tuyến tính trên ma trận
from math_utils import sigmoid, softmax, calLoss # các hàm toán học đã viết ở math_utils.py

#---------Khởi tạo trọng số mạng MLP (1 lớp ẩn)---------
def init_weights(d, h, C):
    # d: số đặc trưng đầu vào (21 cột one-hot của xe ô tô)
    # h: số nơ-ron tại lớp ẩn (ví dụ: 15 node)
    # C: số nhãn đầu ra (4 class: unacc, acc, good, vgood)
    np.random.seed(42)

    # Trọng số nối từ Input (d) sang Hidden (h)
    # Nhân 0.1 để khởi tạo các giá trị ngẫu nhiên nhỏ gần 0, giúp hàm sigmoid không bị bão hòa
    W1 = np.random.randn(h, d) * 0.1      # kích thước (h, d) -> (15, 21)
    b1 = np.zeros((h, 1))                # kích thước (h, 1) -> (15, 1)

    # Trọng số nối từ Hidden (h) sang Output (C)
    W2 = np.random.randn(C, h) * 0.1     # kích thước (C, h) -> (4, 15)
    b2 = np.zeros((C, 1))                # kích thước (C, 1) -> (4, 1)

    return W1, b1, W2, b2


#---------Lan truyền xuôi (Forward Pass)---------
def forward_pass(X, W1, b1, W2, b2):
    # X có kích thước (d, N): mỗi cột là 1 chiếc xe, N là tổng số mẫu
    
    # Bước 1: Đi từ tầng Input vào tầng Hidden
    Z1 = np.dot(W1, X) + b1              # tính tổng tuyến tính lớp ẩn, kích thước (h, N)
    A1 = sigmoid(Z1)                     # đi qua hàm kích hoạt phi tuyến, kích thước (h, N)

    # Bước 2: Đi từ tầng Hidden ra tầng Output
    Z2 = np.dot(W2, A1) + b2             # tính điểm số (logits) cho 4 class, kích thước (C, N)
    A2 = softmax(Z2)                     # đổi điểm số sang xác suất (Y_hat), kích thước (C, N)

    # Trả về toàn bộ các ma trận trung gian để dùng cho bước tính đạo hàm ngược
    return Z1, A1, Z2, A2


#---------Lan truyền ngược (Backward Pass)---------
def backward_pass(X, Y, Z1, A1, Z2, A2, W1, W2):
    N = X.shape[1]                       # số lượng mẫu dữ liệu (1382 mẫu ở tập train)
    C = Y.shape[0]                       # số lượng class (4)
    h = W1.shape[0]                      # số lượng node ẩn (15)

    # 1. Tính độ sai lệch tại tầng Output: dLoss / dZ2
    # Đạo hàm kết hợp của Cross-Entropy và Softmax rút gọn thành (Dự đoán - Nhãn thật)
    dZ2 = A2 - Y                         # kích thước (C, N)

    # 2. Tính đạo hàm cho W2 và b2 tại tầng Output
    dW2 = (1 / N) * np.dot(dZ2, A1.T)    # (C, N) nhân (N, h) -> kích thước (C, h), khớp với W2

    db2 = np.zeros((C, 1))               # kích thước (C, 1), khớp với b2
    for i in range(N):
        for j in range(C):
            db2[j, 0] = db2[j, 0] + dZ2[j, i]
    db2 = db2 / N

    # 3. Truyền ngược sai số từ tầng Output lùi về tầng Hidden
    dA1 = np.dot(W2.T, dZ2)              # (h, C) nhân (C, N) -> kích thước (h, N)
    
    # Đạo hàm của hàm sigmoid(Z1) là: A1 * (1 - A1)
    dZ1 = dA1 * (A1 * (1 - A1))          # nhân từng phần tử (element-wise), kích thước (h, N)

    # 4. Tính đạo hàm cho W1 và b1 tại tầng Hidden
    dW1 = (1 / N) * np.dot(dZ1, X.T)     # (h, N) nhân (N, d) -> kích thước (h, d), khớp với W1

    db1 = np.zeros((h, 1))               # kích thước (h, 1), khớp với b1
    for i in range(N):
        for j in range(h):
            db1[j, 0] = db1[j, 0] + dZ1[j, i]
    db1 = db1 / N

    return dW1, db1, dW2, db2


#---------Huấn luyện mô hình bằng Gradient Descent---------
def train_mlp(X, Y, d, h, C, eta, max_count=5000):
    W1, b1, W2, b2 = init_weights(d, h, C)
    loss_history = []                    # lưu lịch sử Loss để vẽ đồ thị theo dõi

    count = 0
    while count < max_count:
        # Bước 1: Lan truyền xuôi để lấy dự đoán A2
        Z1, A1, Z2, A2 = forward_pass(X, W1, b1, W2, b2)

        # Bước 2: Lan truyền ngược để tính đạo hàm của các trọng số
        dW1, db1, dW2, db2 = backward_pass(X, Y, Z1, A1, Z2, A2, W1, W2)

        # Bước 3: Cập nhật trọng số ngược hướng đạo hàm
        W1 = W1 - eta * dW1
        b1 = b1 - eta * db1
        W2 = W2 - eta * dW2
        b2 = b2 - eta * db2

        # Ghi nhận và in Loss sau mỗi 1000 vòng lặp
        if count % 1000 == 0:
            current_loss = calLoss(A2, Y)
            loss_history.append(current_loss)
            print('Vòng lặp', count, '- Loss trung bình:', round(current_loss, 4))

        count += 1

    return W1, b1, W2, b2, loss_history


#---------Dự đoán nhãn cho dữ liệu mới---------
def predict(X, W1, b1, W2, b2):
    # Chạy lan truyền xuôi để lấy bảng xác suất A2 kích thước (C, N)
    Z1, A1, Z2, A2 = forward_pass(X, W1, b1, W2, b2)

    N = X.shape[1]
    C = A2.shape[0]
    y_pred = np.zeros(N, dtype=int)

    # Duyệt qua từng mẫu xe (từng cột i) để tìm vị trí class có xác suất lớn nhất
    for i in range(N):
        max_val = A2[0, i]
        max_idx = 0
        for j in range(1, C):
            if A2[j, i] > max_val:
                max_val = A2[j, i]
                max_idx = j
        y_pred[i] = max_idx

    return y_pred