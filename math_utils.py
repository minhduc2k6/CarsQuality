#file này để chứa toàn bộ các hàm toán học cần thiết cho việc huấn luyện mạng nơ-ron nhân tạo
import numpy as np # Thư viện tính toán ma trận duy nhất được phép dùng

#---------Các hàm kích hoạt (Activation Functions)---------

def sigmoid(s):
    # Hàm sigmoid dùng cho lớp ẩn (ép giá trị về khoảng 0 đến 1)
    # s có thể là vector hoặc ma trận
    return 1 / (1 + np.exp(-s))


def softmax(s):
    # Hàm softmax dùng cho lớp đầu ra (Output class)
    # s có kích thước (C, N) với C là số class, N là số lượng mẫu
    # np.max(s) giúp trừ đi giá trị lớn nhất trong mỗi cột, ngăn lỗi tràn số học (overflow) khi tính exp()
    e_s = np.exp(s - np.max(s, axis=0, keepdims=True))
    
    # Chia cho tổng của cột để ra xác suất
    return e_s / np.sum(e_s, axis=0, keepdims=True)


#---------Hàm tính độ lỗi (Loss Function)---------

def calLoss(Y_hat, Y):
    # Dùng cho bài toán phân loại đa lớp (Categorical Cross-Entropy)
    # Y_hat: Xác suất dự đoán, kích thước (C, N)
    # Y: Nhãn thật dạng One-hot, kích thước (C, N)
    C = Y_hat.shape[0]
    N = Y_hat.shape[1]
    
    tong = 0
    # Dùng 2 vòng lặp for duyệt qua từng mẫu (N) và từng class (C) đúng bản chất công thức tổng Sigma
    for i in range(N):
        for j in range(C):
            # Cộng thêm 1e-15 vào Y_hat để tránh trường hợp tính log(0) ra NaN
            y_hat_ji = Y_hat[j, i] + 1e-15
            y_ji = Y[j, i]
            
            tong = tong + y_ji * np.log(y_hat_ji)
            
    # Lấy trung bình độ lỗi trên toàn bộ N mẫu
    return -tong / N