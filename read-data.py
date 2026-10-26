import numpy as np
import pandas as pd # đọc CSV, xử lý bảng dữ liệu
import re # thư viện tìm chuỗi theo mẫu (Regular Expression)

train_path = 'raw_data/train.csv'
output_path = 'raw_data/train_extracted.csv'

# Danh sách 24 mẫu xe Nissan để dò tìm (xếp NV200 trước NV để không bắt nhầm)
model_list = [
    'Pathfinder', 'Frontier', 'Altima', 'Armada', 'Maxima', 'Murano',
    'Sentra', 'Xterra', '200SX', '240SX', 'NV200', 'Kicks', 'Quest',
    'Rogue', 'Titan', 'Truck', 'Versa', '350Z', '370Z', 'Cube',
    'GT-R', 'Juke', 'Leaf', 'NV'
]

#---------hàm tách thông số từ 1 dòng Vehicle_Title---------
def extract_car_info(title):
    if not isinstance(title, str):
        return None, None, None, None, None, None

    title = title.strip()

    # 1. Đời xe: lấy 4 ký tự đầu
    year_str = title[:4]
    if year_str.isdigit():
        year = int(year_str)
    else:
        year = None

    # 2. Số cửa: tìm từ có dạng "4dr", "2dr", "3dr"
    door_match = re.search(r'\b(\d+dr)\b', title)
    if door_match:
        doors = door_match.group(1)
    else:
        doors = None

    # 3. Mẫu xe: dò trong danh sách 24 mẫu
    words = title.split() # cắt chuỗi thành từng từ rời nhau
    car_model = None
    for m in model_list:
        if m in words:
            car_model = m
            break

    # Tìm phần nằm trong ngoặc đơn (...) ở cuối câu, ví dụ: (5.6L 8cyl 5A)
    paren_match = re.search(r'\(([^)]+)\)\s*$', title)
    if paren_match:
        inside_paren = paren_match.group(1).strip()
    else:
        inside_paren = ''

    # 4. Dung tích động cơ: tìm từ có dạng "5.6L" trong ngoặc
    disp_match = re.search(r'\b(\d+(?:\.\d+)?L)\b', inside_paren)
    if disp_match:
        displacement = disp_match.group(1)
    else:
        displacement = None

    # 5. Số xi-lanh: tìm từ có dạng "8cyl" trong ngoặc
    cyl_match = re.search(r'\b(\d+cyl)\b', inside_paren)
    if cyl_match:
        cylinders = cyl_match.group(1)
    else:
        cylinders = None

    # 6. Mã hộp số: lấy từ cuối cùng trong ngoặc (ví dụ "5A", "CVT")
    transmission = None
    if inside_paren != '':
        last_word = inside_paren.split()[-1]
        # Bỏ qua nếu trong ngoặc chỉ là năm sản xuất kiểu (1999.5) hoặc (1997.5)
        if last_word not in ['1999.5', '1997.5']:
            transmission = last_word

    return year, car_model, doors, displacement, cylinders, transmission


#---------đọc và xử lý dữ liệu từ CSV---------
def load_and_clean_data(path):
    # truyền lineterminator='\n' để không bị lỗi xuống dòng trong cột Review
    data = pd.read_csv(path, lineterminator='\n')

    # xóa khoảng trắng hoặc ký tự '\r' bị dính ở tên cột
    clean_cols = []
    for col in data.columns:
        clean_cols.append(col.strip())
    data.columns = clean_cols

    # chuyển cột Rating về dạng số thực (float)
    if data['Rating'].dtype == object:
        data['Rating'] = data['Rating'].str.strip().astype(float)

    # tạo các danh sách rỗng để chứa 6 thuộc tính tách ra
    years = []
    models = []
    doors_list = []
    displacements = []
    cylinders_list = []
    transmissions = []

    # lặp qua từng dòng Vehicle_Title để tách thông số
    for title in data['Vehicle_Title']:
        y, m, d, disp, cyl, trans = extract_car_info(title)
        years.append(y)
        models.append(m)
        doors_list.append(d)
        displacements.append(disp)
        cylinders_list.append(cyl)
        transmissions.append(trans)

    # gắn 6 cột mới vào bảng data
    data['Year'] = years
    data['Model'] = models
    data['Doors'] = doors_list
    data['Displacement'] = displacements
    data['Cylinders'] = cylinders_list
    data['Transmission'] = transmissions

    # vứt đi các cột không cần thiết: ID (Unnamed: 0), tên tác giả, tiêu đề và nội dung review
    drop_cols = ['Unnamed: 0', 'Author_Name', 'Review_Title', 'Review']
    data = data.drop(columns=drop_cols)

    return data


#------chạy chương trình và kiểm tra kết quả-----
data_clean = load_and_clean_data(train_path)

print('Kích thước bảng dữ liệu:', data_clean.shape)
print('\n5 dòng đầu tiên sau khi tách:')
print(data_clean.head(5))

print('\nSố lượng ô bị trống (NaN) ở mỗi cột:')
print(data_clean.isnull().sum())

# lưu bảng đã tách ra file CSV mới để dùng cho bước tiếp theo
data_clean.to_csv(output_path, index=False)
print('\nĐã lưu file thành công tại:', output_path)