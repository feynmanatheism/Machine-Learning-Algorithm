# Machine Learning Algorithms Repository

Repository toàn diện để học tập và triển khai các thuật toán Machine Learning với các kiểu học khác nhau.

## 📁 Cấu Trúc Thư Mục

```
Machine Learning Algorithm/
│
├── supervised/                          # Học có giám sát
│   ├── classification/                  # Phân loại (Classification)
│   │   ├── logistic_regression.py
│   │   ├── decision_tree.py
│   │   ├── random_forest.py
│   │   ├── svm.py
│   │   ├── knn.py
│   │   └── naive_bayes.py
│   └── regression/                      # Hồi quy (Regression)
│       ├── linear_regression.py
│       ├── polynomial_regression.py
│       ├── ridge_lasso.py
│       └── svr.py
│
├── unsupervised/                        # Học không giám sát
│   ├── clustering/                      # Phân cụm (Clustering)
│   │   ├── kmeans.py
│   │   ├── hierarchical_clustering.py
│   │   ├── dbscan.py
│   │   └── gaussian_mixture.py
│   └── dimensionality_reduction/        # Giảm chiều
│       ├── pca.py
│       ├── tsne.py
│       └── autoencoder.py
│
├── reinforcement_learning/              # Học tăng cường
│   ├── q_learning.py
│   ├── policy_gradient.py
│   ├── actor_critic.py
│   └── deep_q_learning.py
│
├── deep_learning/                       # Học sâu
│   ├── neural_networks.py
│   ├── convolutional_networks.py
│   ├── recurrent_networks.py
│   └── transformers.py
│
├── utils/                               # Công cụ hữu ích
│   ├── data_preprocessing.py
│   ├── metrics.py
│   └── visualization.py
│
├── examples/                            # Các ví dụ sử dụng
│   ├── supervised_example.py
│   ├── unsupervised_example.py
│   ├── reinforcement_example.py
│   └── deep_learning_example.py
│
├── tests/                               # Các bài kiểm tra
│   ├── test_supervised.py
│   ├── test_unsupervised.py
│   └── test_utils.py
│
├── data/                                # Dữ liệu mẫu
│   └── datasets/
│
├── requirements.txt                     # Dependencies
└── README.md                            # Documentation
```

## 🎯 Các Loại Học (Learning Types)

### 1. **Supervised Learning (Học có giám sát)**
   - **Classification**: Phân loại dữ liệu vào các lớp (Linear Models, Tree-based, SVM, KNN)
   - **Regression**: Dự đoán giá trị liên tục (Linear, Polynomial, Ridge/Lasso, SVR)

### 2. **Unsupervised Learning (Học không giám sát)**
   - **Clustering**: Nhóm các điểm dữ liệu tương tự nhau (K-means, Hierarchical, DBSCAN)
   - **Dimensionality Reduction**: Giảm số lượng feature (PCA, t-SNE, Autoencoders)

### 3. **Reinforcement Learning (Học tăng cường)**
   - Các thuật toán học qua tương tác với môi trường (Q-Learning, Policy Gradient, DQN)

### 4. **Deep Learning (Học sâu)**
   - Các mạng nơ-ron sâu (Neural Networks, CNN, RNN, Transformers)

## 🚀 Cách Sử Dụng

### Cài đặt Dependencies
```bash
pip install -r requirements.txt
```

### Sử dụng một thuật toán
```python
from supervised.classification.logistic_regression import LogisticRegression
from utils.data_preprocessing import load_data, preprocess_data

# Load và chuẩn bị dữ liệu
X, y = load_data('data/your_dataset.csv')
X_train, X_test, y_train, y_test = preprocess_data(X, y)

# Huấn luyện mô hình
model = LogisticRegression()
model.fit(X_train, y_train)

# Dự đoán
predictions = model.predict(X_test)
```

### Chạy các ví dụ
```bash
python examples/supervised_example.py
python examples/unsupervised_example.py
```

## 📊 Các Công Cụ Hữu Ích (Utils)

- **data_preprocessing.py**: Chuẩn bị dữ liệu (normalization, scaling, split)
- **metrics.py**: Tính toán metrics (accuracy, F1, MSE, etc.)
- **visualization.py**: Hình ảnh hóa kết quả (plots, confusion matrix, etc.)

## ✅ Cơ Chế Đặt Tên (Naming Convention)

- File Python: `snake_case` (e.g., `linear_regression.py`)
- Class: `PascalCase` (e.g., `class LinearRegression`)
- Function/Method: `snake_case` (e.g., `def fit()`, `def predict()`)
- Constant: `UPPER_SNAKE_CASE` (e.g., `LEARNING_RATE = 0.01`)

## 📝 Cấu Trúc Mỗi File Thuật Toán

Mỗi file thuật toán nên có:
1. **Docstring**: Mô tả thuật toán
2. **Import**: Các thư viện cần thiết
3. **Class/Function**: Định nghĩa thuật toán
4. **Methods**: `fit()`, `predict()`, `score()` (nếu có)
5. **Example**: Ví dụ sử dụng ở cuối file

```python
"""
Module: Linear Regression
Description: Thuật toán hồi quy tuyến tính cơ bản
Author: Your Name
Date: YYYY-MM-DD
"""

import numpy as np
from typing import Tuple

class LinearRegression:
    """
    Linear Regression Model.
    
    Mô tả thuật toán, công thức toán học, và parameters.
    """
    
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Huấn luyện mô hình"""
        pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Dự đoán"""
        pass

if __name__ == "__main__":
    # Ví dụ sử dụng
    pass
```

## 🔍 Hướng Dẫn Tìm Kiếm Thuật Toán

1. Xác định **loại học** (Supervised/Unsupervised/Reinforcement/Deep)
2. Tìm **danh mục** tương ứng (Classification/Regression/Clustering/etc.)
3. Chọn **file** của thuật toán cần dùng
4. Xem **docstring** và **example** để hiểu cách dùng

## 📚 Tài Liệu Tham Khảo

Mỗi file nên bao gồm:
- Link đến các bài viết giáo dục
- Công thức toán học liên quan
- Reference papers (nếu có)
- Ưu/nhược điểm của thuật toán

## 🤝 Quy Tắc Đóng Góp

Khi thêm thuật toán mới:
1. Tạo file trong thư mục phù hợp
2. Viết docstring chi tiết
3. Thêm examples
4. Cập nhật README.md
5. Viết tests (optional)

---

**Last Updated**: 2026-03-29
