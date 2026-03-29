"""
Example: Using Supervised Learning Algorithms
Module: examples/supervised_example.py

Description:
    Ví dụ hoàn chỉnh về cách sử dụng các thuật toán học có giám sát
"""

import sys
import numpy as np
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import từ các module của project
# from supervised.classification.logistic_regression import LogisticRegression
# from supervised.regression.linear_regression import LinearRegression
# from utils.data_preprocessing import split_data, standardize
# from utils.metrics import accuracy, f1_score, mse


def example_linear_regression():
    """Ví dụ sử dụng Linear Regression"""
    print("=" * 50)
    print("EXAMPLE: Linear Regression")
    print("=" * 50)
    
    # Tạo dữ liệu mẫu
    np.random.seed(42)
    X = np.random.rand(100, 1) * 10
    y = 2.5 * X.squeeze() + 1 + np.random.randn(100) * 2
    
    # Chia dữ liệu
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Ở đây sẽ import và sử dụng LinearRegression từ project
    # model = LinearRegression(learning_rate=0.01, n_iterations=1000)
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)
    # score = model.score(X_test, y_test)
    # print(f"R² Score: {score:.4f}")


def example_classification():
    """Ví dụ sử dụng Classification Algorithms"""
    print("\n" + "=" * 50)
    print("EXAMPLE: Classification with Iris Dataset")
    print("=" * 50)
    
    # Load Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Chia dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"Features: {X.shape[1]}")
    print(f"Classes: {len(np.unique(y))}")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    
    # Ở đây sẽ import và sử dụng classifier từ project
    # from supervised.classification.logistic_regression import LogisticRegression
    # clf = LogisticRegression()
    # clf.fit(X_train, y_train)
    # predictions = clf.predict(X_test)
    # accuracy = np.mean(predictions == y_test)
    # print(f"Accuracy: {accuracy:.4f}")


def example_kmeans_clustering():
    """Ví dụ sử dụng K-Means Clustering"""
    print("\n" + "=" * 50)
    print("EXAMPLE: K-Means Clustering")
    print("=" * 50)
    
    # Tạo dữ liệu có 3 cụm
    np.random.seed(42)
    cluster1 = np.random.randn(30, 2) + [0, 0]
    cluster2 = np.random.randn(30, 2) + [5, 5]
    cluster3 = np.random.randn(30, 2) + [10, 0]
    X = np.vstack([cluster1, cluster2, cluster3])
    
    print(f"Total samples: {X.shape[0]}")
    print(f"Features: {X.shape[1]}")
    
    # Ở đây sẽ import và sử dụng K-Means từ project
    # from unsupervised.clustering.kmeans import KMeans
    # kmeans = KMeans(n_clusters=3, max_iterations=100)
    # kmeans.fit(X)
    # labels = kmeans.predict(X)
    # print(f"Clusters found: {np.unique(labels)}")
    # print(f"Inertia: {kmeans.inertia(X):.4f}")


def workflow_complete():
    """Quy trình hoàn chỉnh: Load -> Preprocess -> Train -> Evaluate"""
    print("\n" + "=" * 50)
    print("COMPLETE WORKFLOW")
    print("=" * 50)
    
    # 1. Load dữ liệu
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=200, n_features=10, n_informative=5,
                               n_redundant=2, n_classes=2, random_state=42)
    print(f"✓ Step 1: Data loaded - Shape: {X.shape}")
    
    # 2. Chia dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"✓ Step 2: Data split - Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
    # 3. Chuẩn bị dữ liệu
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print(f"✓ Step 3: Data preprocessed (standardized)")
    
    # 4. Huấn luyện mô hình
    # from supervised.classification.logistic_regression import LogisticRegression
    # model = LogisticRegression()
    # model.fit(X_train, y_train)
    # print(f"✓ Step 4: Model trained")
    
    # 5. Dự đoán
    # y_pred = model.predict(X_test)
    # print(f"✓ Step 5: Predictions made")
    
    # 6. Đánh giá
    # accuracy = np.mean(y_pred == y_test)
    # print(f"✓ Step 6: Model evaluated - Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    print("\n🚀 Machine Learning Algorithms - Examples\n")
    
    example_linear_regression()
    example_classification()
    example_kmeans_clustering()
    workflow_complete()
    
    print("\n" + "=" * 50)
    print("✅ Tất cả ví dụ hoàn thành!")
    print("=" * 50)
