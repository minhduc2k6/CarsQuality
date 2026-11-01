import numpy as np
import pandas as pd # đọc CSV, mã hóa data
import matplotlib.pyplot as plt # vẽ đồ thị

from sklearn.model_selection import train_test_split, StratifiedKFold
# chia data set thành train/test set và cross-validation 5 lần phân tầng

path = 'car_evaluation.csv'

#---------đọc và mã hóa dữ liệu từ CSV---------
def load_data(path):
    data = pd.read_csv(path)

    feature_names = []

    # mã hóa buying thành 4 cột one-hot (low, med, high, vhigh)
    for val in ['low', 'med', 'high', 'vhigh']:
        col_name = 'is_buying_' + val
        data[col_name] = (data['buying'] == val).astype(int)
        feature_names.append(col_name)

    # mã hóa maint
    for val in ['low', 'med', 'high', 'vhigh']:
        col_name = 'is_maint_' + val
        data[col_name] = (data['maint'] == val).astype(int)
        feature_names.append(col_name)

    # mã hóa doors
    for val in ['2', '3', '4', '5more']:
        col_name = 'is_doors_' + val
        data[col_name] = (data['doors'] == val).astype(int)
        feature_names.append(col_name)

    # mã hóa person
    for val in ['2', '4', 'more']:
        col_name = 'is_persons_' + val
        data[col_name] = (data['persons'] == val).astype(int)
        feature_names.append(col_name)

    # lug_boot 
    for val in ['small', 'med', 'big']:
        col_name = 'is_lug_boot_' + val
        data[col_name] = (data['lug_boot'] == val).astype(int)
        feature_names.append(col_name)

    # safety 
    for val in ['low', 'med', 'high']:
        col_name = 'is_safety_' + val
        data[col_name] = (data['safety'] == val).astype(int)
        feature_names.append(col_name)

    # mã hóa label class thành số 0, 1, 2, 3
    data['class'] = data['class'].map({
        'unacc': 0,
        'acc': 1,
        'good': 2,
        'vgood': 3
    })

    x = data[feature_names].values
    y = data['class'].values
    return x, y, feature_names

x, y, feature_names = load_data(path)

# kiểm tra dòng đầu tiên (vhigh, vhigh, 2, 2, small, low -> [0 0 0 1  0 0 0 1  1 0 0 0  1 0 0  1 0 0  1 0 0])
print('Vector 21 phần tử của dòng đầu tiên:\n', x[0])
print('Kích thước x và y:', x.shape, y.shape)


#------hàm đổi nhãn y (0, 1, 2, 3) sang one-hot 4 cột (dùng khi tính Loss)-----
def onehot_label(y, num_classes=4):
    N = len(y)
    Y_onehot = np.zeros((N, num_classes), dtype=int)

    for i in range(N):
        label = y[i]
        Y_onehot[i, label] = 1

    return Y_onehot


#------chia data thành train/test set (80% train, 20% test)-----
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y)

# tạo nhãn dạng one-hot 4 cột cho tập train và test
Y_train = onehot_label(y_train)
Y_test = onehot_label(y_test)

print('\nKích thước tập Train:', x_train.shape, Y_train.shape)
print('Kích thước tập Test:', x_test.shape, Y_test.shape)


#------chia 80% train bằng Cross-Validation 5 lần phân tầng-----
def get_kfold_splits(x_train, y_train, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = []

    for train_idx, val_idx in skf.split(x_train, y_train):
        x_tr = x_train[train_idx]
        y_tr = y_train[train_idx]
        Y_tr = onehot_label(y_tr)

        x_val = x_train[val_idx]
        y_val = y_train[val_idx]
        Y_val = onehot_label(y_val)

        folds.append((x_tr, y_tr, Y_tr, x_val, y_val, Y_val))

    return folds

folds = get_kfold_splits(x_train, y_train, n_splits=5)
print('Số lượng fold đã chia:', len(folds))
print('Kích thước fold 1 - Train:', folds[0][0].shape, '| Val:', folds[0][3].shape)