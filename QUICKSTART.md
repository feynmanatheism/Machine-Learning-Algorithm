"""
Quickstart Guide - Bắt Đầu Nhanh
Machine Learning Algorithm Repository

Hướng dẫn chi tiết để bắt đầu làm việc với repository
"""

# ============================================================================
# 1. CHUẨN BỊ MÔI TRƯỜNG
# ============================================================================

# Bước 1: Di chuyển vào thư mục dự án
cd "Machine Learning Algorithm"

# Bước 2: Tạo virtual environment
python -m venv venv

# Bước 3: Kích hoạt virtual environment
# Trên Linux/Mac:
source venv/bin/activate

# Hoặc Windows:
# venv\Scripts\activate

# Bước 4: Cập nhật pip
pip install --upgrade pip

# Bước 5: Cài đặt dependencies
pip install -r requirements.txt


# ============================================================================
# 2. CẤU TRÚC CƠ BẢN
# ============================================================================

"""
Cấu trúc thư mục theo loại KIỂU HỌC:

1. SUPERVISED (Học có giám sát) - Có labels
   ├── Classification (Phân loại)
   │   └── logistic_regression.py ✅
   └── Regression (Hồi quy)
       └── linear_regression.py ✅

2. UNSUPERVISED (Học không giám sát) - Không labels
   ├── Clustering (Phân cụm)
   │   └── kmeans.py ✅
   └── Dimensionality Reduction (Giảm chiều)

3. REINFORCEMENT_LEARNING (Học tăng cường)
   └── Q-Learning, Policy Gradient, etc.

4. DEEP_LEARNING (Học sâu)
   └── Neural Networks, CNN, RNN, etc.

5. UTILS (Tiện ích)
   ├── data_preprocessing.py ✅
   ├── metrics.py ✅
   └── visualization.py

6. EXAMPLES (Ví dụ)
   └── supervised_example.py ✅

7. TESTS (Kiểm tra)
   └── test_*.py
"""


# ============================================================================
# 3. SỬ DỤNG THUẬT TOÁN HIỆN CÓ
# ============================================================================

# Ví dụ 1: Linear Regression
from supervised.regression.linear_regression import LinearRegression
from utils.data_preprocessing import split_data, standardize
import numpy as np

# Tạo dữ liệu
X = np.random.rand(100, 5)
y = np.random.rand(100)

# Chia dữ liệu
X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

# Chuẩn hóa dữ liệu
X_train, X_test = standardize(X_train, X_test)

# Huấn luyện
model = LinearRegression(learning_rate=0.01, n_iterations=100)
model.fit(X_train, y_train)

# Dự đoán
predictions = model.predict(X_test)
score = model.score(X_test, y_test)
print(f"R² Score: {score:.4f}")


# Ví dụ 2: K-Means Clustering
from unsupervised.clustering.kmeans import KMeans

# Tạo dữ liệu
X = np.random.rand(100, 2)

# Huấn luyện
kmeans = KMeans(n_clusters=3, max_iterations=100)
kmeans.fit(X)

# Dự đoán
labels = kmeans.predict(X)
inertia = kmeans.inertia(X)
print(f"Inertia: {inertia:.4f}")


# ============================================================================
# 4. THÊM THUẬT TOÁN MỚI
# ============================================================================

# Bước 1: Xác định loại thuật toán
# - Classification? → supervised/classification/
# - Regression? → supervised/regression/
# - Clustering? → unsupervised/clustering/
# - Dimensionality Reduction? → unsupervised/dimensionality_reduction/
# - Reinforcement Learning? → reinforcement_learning/
# - Deep Learning? → deep_learning/

# Bước 2: Tạo file với đúng khuôn mẫu
# File: supervised/classification/decision_tree.py
# Nội dung:

"""
\"\"\"
Decision Tree Classifier
Module: supervised/classification/decision_tree.py

Description:
    Thuật toán cây quyết định để phân loại...
\"\"\"

import numpy as np

class DecisionTree:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
    
    def fit(self, X, y):
        # Implement decision tree
        return self
    
    def predict(self, X):
        # Return predictions
        pass
    
    def score(self, X, y):
        # Calculate accuracy
        pass

if __name__ == "__main__":
    # Example usage
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    
    model = DecisionTree()
    model.fit(X, y)
    predictions = model.predict(X)
    score = model.score(X, predictions)
    print(f"Accuracy: {score:.4f}")
"""

# Bước 3: Chạy và kiểm tra
# python supervised/classification/decision_tree.py


# ============================================================================
# 5. KIỂM TRA VÀ VALIDATION
# ============================================================================

# Chạy ví dụ
python examples/supervised_example.py

# Chạy tests (khi có)
pytest tests/ -v

# Kiểm tra một file cụ thể
python supervised/regression/linear_regression.py


# ============================================================================
# 6. BEST PRACTICES
# ============================================================================

"""
✅ LÀM CÓ:

1. CHUẨN HÓA DỮ LIỆU
   from utils.data_preprocessing import standardize, normalize
   X_train, X_test = standardize(X_train, X_test)

2. CHIA TRAIN/TEST ĐÚNG CÁCH
   from utils.data_preprocessing import split_data
   X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

3. SET RANDOM_STATE (Reproducible)
   model = LinearRegression(random_state=42)

4. VIẾT DOCSTRING CHI TIẾT
   \"\"\"
   Mô tả thuật toán
   
   Args:
       X: Feature matrix
       y: Target vector
   
   Returns:
       Prediction
   \"\"\"

5. KIỂM TRA SHAPE
   print(X.shape, y.shape)

6. SỬ DỤNG TYPE HINTS
   def fit(self, X: np.ndarray, y: np.ndarray) -> 'ModelName'

7. THÊM EXAMPLE SAU CLASS
   if __name__ == "__main__":
       # Example code


❌ KHÔNG LÀM:

1. QUÊN CHIA TRAIN/TEST
   model.fit(X, y)  # ❌ Data leakage!
   model.predict(X)  # ❌ Overfitting!

2. KHÔNG CHUẨN HÓA DỮ LIỆU
   model.fit(X_raw, y)  # ❌ Performance tệ

3. KHÔNG SET RANDOM_STATE
   model = RandomForest()  # ❌ Không reproducible

4. KHÔNG CÓ DOCSTRING
   def fit(self, X, y):  # ❌ Khó hiểu
       pass

5. KHÔNG KIỂM TRA SHAPE
   X_train.fit(X)  # ❌ Có thể lỗi

6. KHÔNG CÓ EXAMPLE
   # ❌ Không biết cách dùng
"""


# ============================================================================
# 7. METRICS - ĐÁNH GIÁ MÔ HÌNH
# ============================================================================

from utils.metrics import (
    accuracy, precision, recall, f1_score,  # Classification
    mean_squared_error, r_squared,           # Regression
    silhouette_score                         # Clustering
)

# Classification Metrics
accuracy_score = accuracy(y_true, y_pred)
precision_score = precision(y_true, y_pred)
recall_score = recall(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# Regression Metrics
mse = mean_squared_error(y_true, y_pred)
r2 = r_squared(y_true, y_pred)

# Clustering Metrics
silhouette = silhouette_score(X, labels)


# ============================================================================
# 8. WORKFLOW HOÀN CHỈNH
# ============================================================================

"""
1. LOAD DỮ LIỆU
   X, y = load_data('data/dataset.csv')

2. EDA (Exploratory Data Analysis)
   print(X.shape, y.shape)
   print(X.describe())

3. CHIA DỮ LIỆU
   X_train, X_test, y_train, y_test = split_data(X, y)

4. CHUẨN BỊ DỮ LIỆU
   X_train, X_test = standardize(X_train, X_test)

5. CHỌN MÔ HÌNH
   from supervised.classification import LogisticRegression
   model = LogisticRegression()

6. HUẤN LUYỆN
   model.fit(X_train, y_train)

7. DỰ ĐOÁN
   y_pred = model.predict(X_test)

8. ĐÁNH GIÁ
   from utils.metrics import accuracy, f1_score
   acc = accuracy(y_test, y_pred)
   f1 = f1_score(y_test, y_pred)
   print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}")

9. TÍA HÓA KẾT QUẢ
   from utils.visualization import plot_results
   plot_results(y_test, y_pred)

10. LƯU MÔ HÌNH (nếu cần)
    import joblib
    joblib.dump(model, 'models/my_model.pkl')
"""


# ============================================================================
# 9. USEFUL COMMANDS
# ============================================================================

# Xem cấu trúc dự án
tree -I 'venv|__pycache__|*.pyc'

# Chạy Python REPL
python

# Chạy Jupyter Notebook
jupyter notebook

# Kiểm tra version package
pip show numpy
pip list

# Cập nhật requirements.txt
pip freeze > requirements.txt

# Xóa __pycache__
find . -type d -name __pycache__ -exec rm -r {} +


# ============================================================================
# 10. TIẾP THEO
# ============================================================================

"""
✨ Bước tiếp theo:

□ Thêm các thuật toán khác:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - SVM
  - PCA
  - Autoencoder

□ Viết comprehensive tests

□ Tạo Jupyter notebooks cho tutorials

□ Thêm benchmark comparisons

□ Tối ưu hóa hiệu suất

□ Tạo documentation hoàn chỉnh

□ Thêm image classification examples

□ Thêm NLP examples

□ Integration với MLflow (experiment tracking)
"""


# ============================================================================
# CÔNG CỤ VÀ THƯ VIỆN HỮUÍCH
# ============================================================================

Libraries:
- numpy: Xử lý mảng
- pandas: Xử lý dữ liệu
- scikit-learn: ML algorithms
- tensorflow/torch: Deep learning
- matplotlib/seaborn: Visualization
- scipy: Scientific computing

Tools:
- Jupyter Notebook: Interactive coding
- VS Code: Code editor
- Git: Version control
- GitHub: Repository hosting
- MLflow: Experiment tracking
- Weights & Biases: ML experiment tracking
- Optuna: Hyperparameter optimization


# ============================================================================
# TÀI LIỆU THAM KHẢO
# ============================================================================

Documentation:
- Scikit-learn: https://scikit-learn.org
- NumPy: https://numpy.org
- Pandas: https://pandas.pydata.org
- TensorFlow: https://tensorflow.org
- PyTorch: https://pytorch.org

Courses:
- Andrew Ng ML Course: coursera.org/learn/machine-learning
- Fast.ai: fast.ai
- Kaggle: kaggle.com/learn

Books:
- Hands-On ML with Scikit-Learn and TensorFlow
- Introduction to Statistical Learning (ISLR)
- Deep Learning Book


---
Happy Learning! 🎉

Nếu có vấn đề, hãy:
1. Kiểm tra docstring của hàm
2. Chạy ví dụ trong if __name__ == "__main__"
3. Đọc error message chi tiết
4. Tìm kiếm trên StackOverflow/GitHub Issues
"""
