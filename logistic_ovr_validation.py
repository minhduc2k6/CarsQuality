
import numpy as np
import pandas as pd

DAC_TRUNG = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]
CAC_LOP = ["unacc", "acc", "good", "vgood"]

ETA = 1.0     
K_FOLD = 5    


GIA_TRI_HOP_LE = {
    "buying":   ["low", "med", "high", "vhigh"],
    "maint":    ["low", "med", "high", "vhigh"],
    "doors":    ["2", "3", "4", "more"],
    "persons":  ["2", "4", "more"],
    "lug_boot": ["small", "med", "big"],
    "safety":   ["low", "med", "high"],
}


# =====================================================================
# BƯỚC 1: CÁC HÀM TOÁN HỌC CỦA HỒI QUY LOGISTIC
# =====================================================================
def sigmoid(s):
    return 1 / (1 + np.exp(-s))


def tinh_J(w, X, y):
    z = sigmoid(X @ w)                    
    eps = 1e-12                          
    return -np.mean(y * np.log(z + eps) + (1 - y) * np.log(1 - z + eps))


def huan_luyen(X, y, eta=0.5, so_vong=2000):
    w = np.zeros(X.shape[1])
    N = len(y)
    for _ in range(so_vong):
        z = sigmoid(X @ w)               
        w = w + eta * (X.T @ (y - z)) / N
    return w


# =====================================================================
# BƯỚC 2: XỬ LÝ DỮ LIỆU
# =====================================================================
def doc_du_lieu():
    
    return pd.read_csv("car_evaluation_imputed.csv", dtype=str)

def ma_hoa(df):
   
    cot = [f"{c}_{v}" for c in DAC_TRUNG for v in GIA_TRI_HOP_LE[c]]
    one_hot = pd.get_dummies(df[DAC_TRUNG].astype(str))
    return one_hot.reindex(columns=cot, fill_value=0).astype(float).values





# =====================================================================
# BƯỚC 3: MÔ HÌNH (One-vs-Rest: mỗi lớp 1 mô hình nhị phân)
# =====================================================================
class MoHinh:
    def fit(self, X_tho, y, eta=0.5, in_J=True):
        self.mean = X_tho.mean(axis=0)
        self.std = X_tho.std(axis=0) + 1e-8
        X = self._chuan_bi(X_tho)

        self.w = []
        for k, ten in enumerate(CAC_LOP):
            y_k = (y == k).astype(float)       
            w_k = huan_luyen(X, y_k, eta)
            self.w.append(w_k)
            if in_J:
                print(f"  Lớp {ten:6s}: J(w) = {tinh_J(w_k, X, y_k):.4f}")
        self.w = np.array(self.w)              

    def _chuan_bi(self, X_tho):
        X = (X_tho - self.mean) / self.std
        return np.hstack([np.ones((len(X), 1)), X])        

    def xac_suat(self, X_tho):
        X = self._chuan_bi(X_tho)
        s = X @ self.w.T                        
        return sigmoid(s)                      

    def du_doan(self, X_tho):
        return self.xac_suat(X_tho).argmax(axis=1)


# =====================================================================
# BƯỚC 4: ĐÁNH GIÁ
# =====================================================================
def danh_gia(mo_hinh, X, y, ten):
    pred = mo_hinh.du_doan(X)
    print(f"\n[{ten}] Độ chính xác: {(pred == y).mean():.4f}")

    cm = np.zeros((4, 4), dtype=int)
    for that, doan in zip(y, pred):
        cm[that, doan] += 1
    print("Ma trận nhầm lẫn (hàng = thực tế, cột = dự đoán):")
    print(pd.DataFrame(cm, index=CAC_LOP, columns=CAC_LOP))

    print("Precision / Recall / F1-score:")
    f1_cac_lop = []
    for k, lop in enumerate(CAC_LOP):
        prec = cm[k, k] / cm[:, k].sum() if cm[:, k].sum() else 0
        rec = cm[k, k] / cm[k].sum() if cm[k].sum() else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0   # F1 = 2PR/(P+R)
        f1_cac_lop.append(f1)
        print(f"  {lop:6s} precision={prec:.3f}  recall={rec:.3f}  f1={f1:.3f}")

    so_mau = cm.sum(axis=1)
    macro_f1 = np.mean(f1_cac_lop)
    weighted_f1 = np.sum(np.array(f1_cac_lop) * so_mau) / so_mau.sum()
    print(f"  Macro F1-score    = {macro_f1:.4f}")
    print(f"  Weighted F1-score = {weighted_f1:.4f}")
    return macro_f1


def in_bo_trong_so(mo_hinh):
    ten_hang = ["Intercept (Bias w0)"] + [f"{c}_{v}" for c in DAC_TRUNG for v in GIA_TRI_HOP_LE[c]]
    rong = max(len(t) for t in ten_hang)           # độ rộng cột tên để các dấu ":" thẳng hàng
    for k, lop in enumerate(CAC_LOP):
        print(f"\n=== BỘ TRỌNG SỐ (mô hình lớp {lop}) ===")
        for ten, gia_tri in zip(ten_hang, mo_hinh.w[k]):
            print(f"{ten:<{rong}} : {gia_tri:+.6f}")


# =====================================================================
# BƯỚC 5: HUẤN LUYỆN BẰNG K-FOLD VÀ CHỌN MÔ HÌNH
# =====================================================================
def chay_huan_luyen(in_log=True):
    df = doc_du_lieu()
    X = ma_hoa(df)
    y = df["class"].map({c: i for i, c in enumerate(CAC_LOP)}).values

    
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    n_train_full = int(0.8 * len(X))
    train_full, test = idx[:n_train_full], idx[n_train_full:]
    if in_log:
        print(f"Số mẫu train_full/test: {len(train_full)}/{len(test)}")

   
    cac_phan = np.array_split(train_full, K_FOLD)
    acc_train_k, acc_val_k = [], []
    cac_mo_hinh = []                           
    if in_log:
        print(f"\n{K_FOLD}-fold cross-validation (η = {ETA}):")
    for i in range(K_FOLD):
        val_i = cac_phan[i]
        train_i = np.concatenate([cac_phan[j] for j in range(K_FOLD) if j != i])
        M_i = MoHinh()
        M_i.fit(X[train_i], y[train_i], eta=ETA, in_J=False)
        cac_mo_hinh.append(M_i)
        a_tr = (M_i.du_doan(X[train_i]) == y[train_i]).mean()
        a_va = (M_i.du_doan(X[val_i]) == y[val_i]).mean()
        acc_train_k.append(a_tr)
        acc_val_k.append(a_va)
        if in_log:
            print(f"  Lần {i + 1}: validation = X{i + 1} ({len(val_i)} xe), training = {len(train_i)} xe"
                  f" | acc train = {a_tr:.4f} | acc validation = {a_va:.4f}")
    if in_log:
        print(f"  Trung bình {K_FOLD} lần: acc train = {np.mean(acc_train_k):.4f}"
              f" | acc validation = {np.mean(acc_val_k):.4f}"
              f" (độ lệch chuẩn {np.std(acc_val_k):.4f})")

    sai_so_val = 1 - np.array(acc_val_k)
    i_tot_nhat = int(np.argmin(sai_so_val))
    mo_hinh = cac_mo_hinh[i_tot_nhat]
    if in_log:
        print(f"\nValidation error của {K_FOLD} mô hình: " +
              "  ".join(f"M{i + 1}={e:.4f}" for i, e in enumerate(sai_so_val)))
        print(f"-> Chọn M{i_tot_nhat + 1} (validation error nhỏ nhất = {sai_so_val[i_tot_nhat]:.4f})")
        train_chon = np.concatenate([cac_phan[j] for j in range(K_FOLD) if j != i_tot_nhat])
        danh_gia(mo_hinh, X[train_chon], y[train_chon], f"Train của M{i_tot_nhat + 1}")
        danh_gia(mo_hinh, X[cac_phan[i_tot_nhat]], y[cac_phan[i_tot_nhat]], f"Validation của M{i_tot_nhat + 1}")
        danh_gia(mo_hinh, X[test], y[test], "Test")
    return mo_hinh


# =====================================================================
# BƯỚC 6: DỰ ĐOÁN XE MỚI (NHẬP BẰNG TAY)
# =====================================================================
def in_ket_qua(mo_hinh, df_moi):
    zs = mo_hinh.xac_suat(ma_hoa(df_moi))
    for i, (_, xe) in enumerate(df_moi.iterrows()):
        k = zs[i].argmax()
        chi_tiet = "  ".join(f"{l}={p:.2f}" for l, p in zip(CAC_LOP, zs[i]))
        print(f"{' | '.join(str(xe[c]) for c in DAC_TRUNG)}  ->  {CAC_LOP[k]}   ({chi_tiet})")


def nhap_xe_bang_tay(mo_hinh):
    print("\nNhập thông số xe (Enter trống để thoát).")
    while True:
        xe = {}
        for c in DAC_TRUNG:
            ok = GIA_TRI_HOP_LE[c]
            while True:
                v = input(f"  {c} {ok}: ").strip()
                if v == "" and c == DAC_TRUNG[0]:
                    return
                if v in ok:
                    xe[c] = v
                    break
                print("   -> Giá trị không hợp lệ, nhập lại.")
        in_ket_qua(mo_hinh, pd.DataFrame([xe]))
        print()


# =====================================================================
# CHẠY CHƯƠNG TRÌNH
# =====================================================================
if __name__ == "__main__":
    mo_hinh = chay_huan_luyen()
    in_bo_trong_so(mo_hinh)

    
    while True:
        print("\n===== THỬ DỮ LIỆU MỚI =====")
        print("1. Nhập 1 chiếc xe bằng tay")
        print("0. Thoát")
        chon = input("Chọn: ").strip()
        if chon == "1":
            nhap_xe_bang_tay(mo_hinh)
        else:
            break