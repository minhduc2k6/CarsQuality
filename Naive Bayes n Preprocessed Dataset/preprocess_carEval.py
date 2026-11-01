import itertools
import os
import sys
import time
import numpy as np
import pandas as pd

# ================= CẤU HÌNH =================
INPUT_CSV = sys.argv[1] if len(sys.argv) > 1 else "car_evaluation_missing.csv"
OUTPUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "."
OUTPUT_NAME = "car_evaluation_cleaned.csv"
TARGET = "class"

# Định nghĩa các giá trị hợp lệ theo đúng thứ tự gốc của bộ dữ liệu
FILE_ORDER = {
    "buying":   ["vhigh", "high", "med", "low"],
    "maint":    ["vhigh", "high", "med", "low"],
    "doors":    ["2", "3", "4", "5more"],
    "persons":  ["2", "4", "more"],
    "lug_boot": ["small", "med", "big"],
    "safety":   ["low", "med", "high"],
}
CLASS_ORDER = ["unacc", "acc", "good", "vgood"]
MISSING_TOKENS = ["", "?", "nan", "na", "n/a", "none", "null", "-"]


# ================= CÁC BƯỚC TIỀN XỬ LÝ =================

def step1_load(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = list(FILE_ORDER.keys()) + [TARGET]
    
    print(f"[Bước 1] Đọc xong: {df.shape[0]} dòng, {df.shape[1]} cột.")
    return df[required]


def step2_normalize_text(df):
    df = df.copy()
    for col in df.columns:
        s = df[col].astype(str).str.strip().str.lower()
        s = s.replace(MISSING_TOKENS, np.nan)
        df[col] = s
    print("[Bước 2] Đã chuẩn hóa văn bản và đưa giá trị thiếu về chuẩn NaN.")
    return df


def step3_invalid_to_nan(df):
    df = df.copy()
    valid_map = {**FILE_ORDER, TARGET: CLASS_ORDER}
    
    for col, valid_values in valid_map.items():
        invalid_mask = df[col].notna() & ~df[col].isin(valid_values)
        df.loc[invalid_mask, col] = np.nan
        
    print("[Bước 3] Đã lọc các giá trị không hợp lệ thành NaN.")
    return df


def step4_impute_features(df):
    df = df.copy()
    features = list(FILE_ORDER.keys())
    
    design = pd.DataFrame(
        list(itertools.product(*FILE_ORDER.values())), 
        columns=features
    )
    # design: Đầy đủ 1728 bản ghi
    
    use_design = False
    if len(df) == len(design):
        matched = (df[features] == design) & df[features].notna()
        total_known = df[features].notna().sum().sum()
        rate = matched.sum().sum() / total_known if total_known > 0 else 0
        
        print(f"        Độ khớp thứ tự dòng gốc: {rate:.2%}")
        if rate >= 0.999:
            use_design = True
            
    if use_design:
        df[features] = df[features].fillna(design)
        print("[Bước 4] Điền thiếu: Khôi phục chính xác theo vị trí dòng.")
    else:
        for col in features:
            mode_val = df[col].mode()
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val.iloc[0])
        print("[Bước 4] Điền thiếu: Dùng mode của từng cột (do thứ tự bị xáo trộn).")
        
    return df


def step5_remove_missing_target(df):
    before = len(df)
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    print(f"[Bước 5] Đã xóa {before - len(df)} dòng thiếu nhãn '{TARGET}'.")
    return df


def step6_check_duplicates(df):
    n_dup = df.duplicated().sum()
    print(f"[Bước 6] Số dòng trùng lặp hoàn toàn: {n_dup} (giữ nguyên).")
    return df


def step7_save(df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
    
    try:
        df.to_csv(path, index=False)
    except PermissionError:
        path = os.path.join(OUTPUT_DIR, f"car_evaluation_cleaned_{time.strftime('%H%M%S')}.csv")
        df.to_csv(path, index=False)
        print("        File đang bị khóa, đã lưu sang tên mới.")
        
    print(f"[Bước 7] Đã lưu file thành công tại: {path}")
    print("        Phân bố nhãn:", df[TARGET].value_counts().to_dict())


# ================= CHƯƠNG TRÌNH CHÍNH =================
def main():
    df = step1_load(INPUT_CSV)
    df = step2_normalize_text(df)
    df = step3_invalid_to_nan(df)
    df = step4_impute_features(df)
    df = step5_remove_missing_target(df)
    df = step6_check_duplicates(df)
    step7_save(df)


if __name__ == "__main__":
    main()