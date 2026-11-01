import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, train_test_split

# ============================ CẤU HÌNH ============================
INPUT_CSV = sys.argv[1] if len(sys.argv) > 1 else "car_evaluation_cleaned.csv"
TARGET = "class"
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5
ALPHA = 1.0
TUNE_ALPHA = True
ALPHA_GRID = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
CLASS_ORDER = ["unacc", "acc", "good", "vgood"]


# ============================ MÔ HÌNH ============================
class CategoricalNaiveBayes:
    def __init__(self, classes, categories, alpha=1.0):
        self.classes = list(classes)
        self.categories = categories
        self.alpha = alpha

    def fit(self, X, y):
        X, y = X.reset_index(drop=True), pd.Series(y).reset_index(drop=True)
        k = len(self.classes)

        # Tiên nghiệm P(lớp)
        counts = y.value_counts().reindex(self.classes).fillna(0).to_numpy(float)
        self.log_prior_ = np.log(counts / counts.sum()) #Prior dạng logarit

        # Hợp lý P(giá trị | lớp) với làm mượt Laplace
        self.features_ = list(X.columns)
        self.log_lik_, self.cat_index_ = {}, {}
        for col in self.features_:
            cats = self.categories[col]
            table = pd.crosstab(y, X[col]).reindex(index=self.classes, columns=cats, fill_value=0).to_numpy(float)
            smoothed = (table + self.alpha) / (table.sum(axis=1, keepdims=True) + self.alpha * len(cats)) #Them alpha tranh P = 0
            self.log_lik_[col] = np.log(smoothed)
            self.cat_index_[col] = pd.Index(cats)
        return self

    def predict_proba(self, X):
        X = X.reset_index(drop=True)
        n, k = len(X), len(self.classes)
        scores = np.tile(self.log_prior_, (n, 1)) # n dong self.log_prior
        
        for col in self.features_:
            idx = self.cat_index_[col].get_indexer(X[col]) #tim index tung attribute
            known = idx >= 0
            contrib = np.zeros((n, k)) #dung de xu ly gia tri la
            contrib[known] = self.log_lik_[col][:, idx[known]].T #1 hang (gtri) & 4 cot (lop) (True car only)
            scores += contrib

        scores -= scores.max(axis=1, keepdims=True) # Chuẩn hóa số học
        proba = np.exp(scores)
        return proba / proba.sum(axis=1, keepdims=True)

    def predict(self, X):
        return np.array(self.classes, dtype=object)[self.predict_proba(X).argmax(axis=1)] #return nhan theo Idx co XS cao nhat


# ============================ 4 BƯỚC XỬ LÝ ============================

def step1_load(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False) #kieu String
    if TARGET not in df.columns or (df == "").any().any():
        sys.exit(f"Lỗi: File thiếu cột '{TARGET}' hoặc chứa ô trống!") #Check xem qua preprocess chua
    print(f"[Bước 1] Đọc dữ liệu thành công: {df.shape[0]} dòng, {df.shape[1]} cột.")
    return df

def step2_split(df):
    X, y = df.drop(columns=[TARGET]), df[TARGET] #X=features, y=classes
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    print(f"[Bước 2] Chia dữ liệu: Train={len(X_train)}, Test={len(X_test)}")
    return X_train, X_test, y_train, y_test

def step3_train(X_train, y_train, classes, categories):
    n_splits = int(min(CV_FOLDS, y_train.value_counts().min())) #n_splits = min( 5 , count(vgood) )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    alphas = ALPHA_GRID if TUNE_ALPHA else [ALPHA] #TRUE = Tim ALPHA tot hon 1.0 neu co

    # Tìm alpha tối ưu qua CV
    best_alpha, best_f1 = ALPHA, -1
    cv_scores = {}
    for alpha in alphas: #alphas = [1.0]
        f1_list = [] #Chua score de tinh mean
        for tr, va in cv.split(X_train, y_train): #Chia train (new) & validate
            m = CategoricalNaiveBayes(classes, categories, alpha).fit(X_train.iloc[tr], y_train.iloc[tr])
            p = m.predict(X_train.iloc[va])
            f1_list.append(precision_recall_fscore_support(y_train.iloc[va], p, average="macro", zero_division=0)[2]) #Them F1Score, z_d=0 tranh crash
        mean_f1 = np.mean(f1_list)
        if mean_f1 > best_f1:
            best_f1, best_alpha = mean_f1, alpha
        cv_scores[alpha] = f1_list

    print(f"[Bước 3] Huấn luyện xong với alpha tốt nhất = {best_alpha} (F1 macro CV = {best_f1:.4f})")
    model = CategoricalNaiveBayes(classes, categories, best_alpha).fit(X_train, y_train)
    return model, n_splits, np.mean(cv_scores[best_alpha]), np.std(cv_scores[best_alpha])

def step4_evaluate(model, X_train, X_test, y_train, y_test, n_splits, cv_mean, cv_std):
    pred = model.predict(X_test)
    y_true = y_test.to_numpy()

    acc = accuracy_score(y_true, pred)
    bal = balanced_accuracy_score(y_true, pred)
    p_mac, r_mac, f_mac, _ = precision_recall_fscore_support(y_true, pred, average="macro", zero_division=0)
    train_acc = accuracy_score(y_train, model.predict(X_train))

    print("\n" + "=" * 50)
    print("KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST")
    print("=" * 50)
    print(f"Accuracy         : {acc:.4f}")
    print(f"Balanced Accuracy: {bal:.4f}")
    print(f"Precision/Recall/F1 (macro): {p_mac:.4f} / {r_mac:.4f} / {f_mac:.4f}")
    print(f"Train Accuracy   : {train_acc:.4f}  (so sánh để check overfit)")
    print(f"CV F1-macro ({n_splits}-fold): {cv_mean:.4f} ± {cv_std:.4f}")
    print("-" * 50)
    print("BÁO CÁO CHI TIẾT:")
    print(classification_report(y_true, pred, labels=model.classes, zero_division=0, digits=4))
    
    cm = pd.DataFrame(confusion_matrix(y_true, pred, labels=model.classes),
                      index=[f"Thực tế: {c}" for c in model.classes], 
                      columns=[f"Dự đoán: {c}" for c in model.classes])
    print("MA TRẬN NHẦM LẪN:")
    print(cm.to_string())


# ============================ CHƯƠNG TRÌNH CHÍNH ============================
def main():
    df = step1_load(INPUT_CSV)
    present = set(df[TARGET])
    classes = [c for c in CLASS_ORDER if c in present] + sorted(present - set(CLASS_ORDER))
    categories = {c: sorted(df[c].unique()) for c in df.columns if c != TARGET}

    X_train, X_test, y_train, y_test = step2_split(df)
    model, n_splits, cv_mean, cv_std = step3_train(X_train, y_train, classes, categories)
    step4_evaluate(model, X_train, X_test, y_train, y_test, n_splits, cv_mean, cv_std)

if __name__ == "__main__":
    main()