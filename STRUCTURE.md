# Machine Learning Algorithm Repository

## 🎯 Cấu Trúc Dự Án - Project Structure

```
Machine Learning Algorithm/
│
├── 📄 README.md                           # Tài liệu chính
├── 📄 requirements.txt                    # Dependencies
├── .gitignore                             # Git ignore rules
│
├── 📁 supervised/                         # Học có giám sát
│   ├── classification/                    # Phân loại
│   │   ├── logistic_regression.py        # ✅ Logistic Regression
│   │   ├── decision_tree.py              # [ ] Decision Tree
│   │   ├── random_forest.py              # [ ] Random Forest
│   │   ├── svm.py                        # [ ] Support Vector Machine
│   │   ├── knn.py                        # [ ] K-Nearest Neighbors
│   │   └── naive_bayes.py                # [ ] Naive Bayes
│   │
│   └── regression/                        # Hồi quy
│       ├── linear_regression.py          # ✅ Linear Regression
│       ├── polynomial_regression.py      # [ ] Polynomial Regression
│       ├── ridge_lasso.py                # [ ] Ridge/Lasso Regression
│       └── svr.py                        # [ ] Support Vector Regression
│
├── 📁 unsupervised/                       # Học không giám sát
│   ├── clustering/                        # Phân cụm
│   │   ├── kmeans.py                     # ✅ K-Means
│   │   ├── hierarchical_clustering.py    # [ ] Hierarchical Clustering
│   │   ├── dbscan.py                     # [ ] DBSCAN
│   │   └── gaussian_mixture.py           # [ ] Gaussian Mixture Model
│   │
│   └── dimensionality_reduction/          # Giảm chiều
│       ├── pca.py                        # [ ] Principal Component Analysis
│       ├── tsne.py                       # [ ] t-SNE
│       └── autoencoder.py                # [ ] Autoencoder
│
├── 📁 reinforcement_learning/             # Học tăng cường
│   ├── q_learning.py                     # [ ] Q-Learning
│   ├── policy_gradient.py                # [ ] Policy Gradient
│   ├── actor_critic.py                   # [ ] Actor-Critic
│   └── deep_q_learning.py                # [ ] Deep Q-Network (DQN)
│
├── 📁 deep_learning/                      # Học sâu
│   ├── neural_networks.py                # [ ] Basic Neural Networks
│   ├── convolutional_networks.py         # [ ] CNN (Convolutional Neural Networks)
│   ├── recurrent_networks.py             # [ ] RNN/LSTM
│   └── transformers.py                   # [ ] Transformer Models
│
├── 📁 utils/                              # Công cụ hữu ích
│   ├── data_preprocessing.py             # ✅ Xử lý dữ liệu
│   ├── metrics.py                        # [ ] Các metric đánh giá
│   └── visualization.py                  # [ ] Hình ảnh hóa
│
├── 📁 examples/                           # Ví dụ sử dụng
│   ├── supervised_example.py             # ✅ Ví dụ Supervised Learning
│   ├── unsupervised_example.py           # [ ] Ví dụ Unsupervised Learning
│   ├── reinforcement_example.py          # [ ] Ví dụ Reinforcement Learning
│   └── deep_learning_example.py          # [ ] Ví dụ Deep Learning
│
├── 📁 tests/                              # Kiểm tra
│   ├── test_supervised.py                # [ ] Test Supervised Algorithms
│   ├── test_unsupervised.py              # [ ] Test Unsupervised Algorithms
│   └── test_utils.py                     # [ ] Test Utilities
│
└── 📁 data/                               # Dữ liệu mẫu
    └── datasets/                         # Thư mục lưu dữ liệu
```

## 📊 Hướng Dẫn Sử Dụng Nhanh

### 1️⃣ Khởi tạo Project

```bash
cd "Machine Learning Algorithm"
pip install -r requirements.txt
```

### 2️⃣ Cấu Trúc của Mỗi File Thuật Toán

Mỗi file thuật toán nên tuân theo cấu trúc:

```python
"""
Algorithm Name
Module: path/to/algorithm.py
Description: Chi tiết về thuật toán
"""

import numpy as np
from typing import Tuple

class AlgorithmName:
    """Class docstring"""
    
    def __init__(self, **params):
        """Khởi tạo parameters"""
        pass
    
    def fit(self, X, y):
        """Huấn luyện mô hình"""
        pass
    
    def predict(self, X):
        """Dự đoán"""
        pass
    
    def score(self, X, y):
        """Đánh giá mô hình"""
        pass

if __name__ == "__main__":
    # Example usage
    pass
```

### 3️⃣ Sử Dụng Một Thuật Toán

```python
# Import
from supervised.regression.linear_regression import LinearRegression
from utils.data_preprocessing import split_data

# Load & prepare data
X, y = load_your_data()
X_train, X_test, y_train, y_test = split_data(X, y)

# Create & train model
model = LinearRegression(learning_rate=0.01)
model.fit(X_train, y_train)

# Predict & evaluate
predictions = model.predict(X_test)
score = model.score(X_test, y_test)
print(f"R² Score: {score}")
```

### 4️⃣ Chạy Ví Dụ

```bash
# Run supervised learning examples
python examples/supervised_example.py

# Run unsupervised learning examples
python examples/unsupervised_example.py
```

## 📋 Quy Tắc Đặt Tên

| Loại | Quy Tắc | Ví Dụ |
|------|---------|--------|
| File | `snake_case` | `linear_regression.py` |
| Class | `PascalCase` | `class LinearRegression` |
| Function | `snake_case` | `def fit()`, `def predict()` |
| Constant | `UPPER_SNAKE_CASE` | `LEARNING_RATE = 0.01` |
| Variable | `snake_case` | `n_features`, `X_train` |

## 🎓 Các Bước Thêm Thuật Toán Mới

1. **Chọn thư mục phù hợp**
   - Supervised/Unsupervised/Reinforcement/Deep Learning
   - Classification/Regression/Clustering/etc.

2. **Tạo file mới với cấu trúc chuẩn**
   ```bash
   touch supervised/classification/my_algorithm.py
   ```

3. **Viết thuật toán**
   - Docstring chi tiết
   - Thực thi chính
   - Example usage

4. **Cập nhật `__init__.py`** của thư mục cha

5. **Viết tests** (optional)

6. **Cập nhật README.md**

## 🧪 Chạy Tests

```bash
pytest tests/
pytest tests/test_supervised.py -v
```

## 📚 Tài Liệu và Tham Khảo

### Recommended Learning Resources
- [Scikit-learn Documentation](https://scikit-learn.org)
- [ML Mastery](https://machinelearningmastery.com/)
- [Andrew Ng's ML Course](https://www.coursera.org/learn/machine-learning)

### Key Concepts
- **Supervised Learning**: Dữ liệu có labels
- **Unsupervised Learning**: Dữ liệu không có labels
- **Reinforcement Learning**: Học qua trial-and-error
- **Deep Learning**: Các mạng nơ-ron sâu

## 💡 Tips & Best Practices

✅ **Nên làm:**
- Viết docstring chi tiết
- Chuẩn hóa dữ liệu trước huấn luyện
- Chia train/test set đúng cách
- Kiểm tra với ví dụ đơn giản trước

❌ **Không nên:**
- Không chuẩn hóa dữ liệu
- Train/test splitting không đúng (data leakage)
- Quên set random_state (không reproducible)
- Không validate mô hình

## 🚀 Tiếp Theo

- Thêm thuật toán theo nhu cầu
- Tối ưu hóa hiệu suất
- Viết comprehensive tests
- Tạo Jupyter notebooks cho tutorials

---

**Repository Updated**: 29 March 2026
**Status**: ✅ Setup Complete - Ready for Implementation
