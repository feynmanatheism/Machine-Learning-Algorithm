"""
K-Means Clustering - Phân Cụm K-Means
Module: unsupervised/clustering/kmeans.py

Description:
    Thuật toán phân cụm K-Means - phân chia dữ liệu thành K cụm.
    Sử dụng centroid (trung tâm) để đại diện cho mỗi cụm.
    
    Quy trình:
    1. Khởi tạo K centroid ngẫu nhiên
    2. Gán mỗi điểm vào centroid gần nhất
    3. Cập nhật centroid mới bằng trung bình các điểm
    4. Lặp lại bước 2-3 đến khi hội tụ

Author: Your Name
Date: 2026-03-29
"""

import numpy as np
from typing import Tuple


class KMeans:
    """
    K-Means Clustering Model
    
    Attributes:
        n_clusters (int): Số cụm K
        max_iterations (int): Số lần lặp tối đa
        random_state (int): Seed cho khởi tạo ngẫu nhiên
        centroids (np.ndarray): Tâm của mỗi cụm
        labels (np.ndarray): Nhãn cụm của mỗi mẫu
    """
    
    def __init__(self, n_clusters: int = 3, max_iterations: int = 100, random_state: int = 42):
        """
        Khởi tạo K-Means
        
        Args:
            n_clusters (int): Số cụm
            max_iterations (int): Số lần lặp tối đa
            random_state (int): Seed ngẫu nhiên
        """
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.centroids = None
        self.labels = None
    
    def _euclidean_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Tính khoảng cách Euclidean giữa hai điểm"""
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """Khởi tạo K centroid ngẫu nhiên từ dữ liệu"""
        np.random.seed(self.random_state)
        random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        return X[random_indices]
    
    def fit(self, X: np.ndarray) -> 'KMeans':
        """
        Huấn luyện mô hình K-Means
        
        Args:
            X (np.ndarray): Feature matrix, shape (n_samples, n_features)
        
        Returns:
            KMeans: Mô hình đã được huấn luyện
        """
        # Khởi tạo centroids
        self.centroids = self._initialize_centroids(X)
        
        for iteration in range(self.max_iterations):
            # Gán mỗi điểm vào centroid gần nhất
            distances = np.zeros((X.shape[0], self.n_clusters))
            for i, centroid in enumerate(self.centroids):
                distances[:, i] = np.array([self._euclidean_distance(x, centroid) for x in X])
            
            self.labels = np.argmin(distances, axis=1)
            
            # Cập nhật centroids
            new_centroids = np.zeros_like(self.centroids)
            for i in range(self.n_clusters):
                cluster_points = X[self.labels == i]
                if len(cluster_points) > 0:
                    new_centroids[i] = cluster_points.mean(axis=0)
                else:
                    new_centroids[i] = self.centroids[i]
            
            # Kiểm tra hội tụ
            if np.allclose(self.centroids, new_centroids):
                print(f"K-Means hội tụ sau {iteration + 1} lần lặp")
                break
            
            self.centroids = new_centroids
            
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iterations}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Dự đoán cụm cho dữ liệu mới
        
        Args:
            X (np.ndarray): Feature matrix
        
        Returns:
            np.ndarray: Nhãn cụm của mỗi mẫu
        """
        distances = np.zeros((X.shape[0], self.n_clusters))
        for i, centroid in enumerate(self.centroids):
            distances[:, i] = np.array([self._euclidean_distance(x, centroid) for x in X])
        
        return np.argmin(distances, axis=1)
    
    def inertia(self, X: np.ndarray) -> float:
        """
        Tính inertia (tổng bình phương khoảng cách từ mỗi mẫu đến centroid)
        Giá trị thấp hơn = cụm compactness tốt hơn
        """
        distances = np.zeros(X.shape[0])
        for i, centroid in enumerate(self.centroids):
            cluster_points = X[self.labels == i]
            distances[self.labels == i] = np.array(
                [self._euclidean_distance(x, centroid) for x in cluster_points]
            )
        return np.sum(distances ** 2)
    
    def get_centroids(self) -> np.ndarray:
        """Trả về tâm của các cụm"""
        return self.centroids


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Tạo dữ liệu mẫu
    np.random.seed(42)
    
    # 3 cụm
    cluster1 = np.random.randn(30, 2) + [0, 0]
    cluster2 = np.random.randn(30, 2) + [5, 5]
    cluster3 = np.random.randn(30, 2) + [10, 0]
    
    X = np.vstack([cluster1, cluster2, cluster3])
    
    # Huấn luyện K-Means
    kmeans = KMeans(n_clusters=3, max_iterations=100)
    kmeans.fit(X)
    
    # Dự đoán
    labels = kmeans.predict(X)
    
    # Kết quả
    print(f"Inertia: {kmeans.inertia(X):.4f}")
    print(f"Centroids shape: {kmeans.get_centroids().shape}")
    print(f"Unique clusters: {np.unique(labels)}")
