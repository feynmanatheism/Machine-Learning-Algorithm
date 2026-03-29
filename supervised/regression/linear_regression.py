"""
Linear Regression - Hồi Quy Tuyến Tính
Module: supervised/regression/linear_regression.py

Description:
    Thuật toán hồi quy tuyến tính cơ bản.
    Tìm mối quan hệ tuyến tính giữa biến độc lập (X) và biến phụ thuộc (y).
    
    Công thức: y = w*X + b
    Mục tiêu: Tối thiểu hóa sai số bình phương trung bình (MSE)

Author: Your Name
Date: 2026-03-29
"""

import numpy as np
from typing import Tuple, Union
from abc import ABC, abstractmethod


class LinearRegression:
    """
    Linear Regression Model
    
    Attributes:
        learning_rate (float): Tốc độ học (mặc định: 0.01)
        n_iterations (int): Số lần lặp training (mặc định: 1000)
        weights (np.ndarray): Trọng số của mô hình
        bias (float): Độ lệch (bias)
        cost_history (list): Lịch sử giá trị loss theo từng epoch
    """
    
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        """
        Khởi tạo Linear Regression
        
        Args:
            learning_rate (float): Tốc độ học
            n_iterations (int): Số lần lặp
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = 0
        self.cost_history = []
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """
        Huấn luyện mô hình
        
        Args:
            X (np.ndarray): Feature matrix, shape (n_samples, n_features)
            y (np.ndarray): Target vector, shape (n_samples,)
        
        Returns:
            LinearRegression: Mô hình đã được huấn luyện
        """
        n_samples, n_features = X.shape
        
        # Khởi tạo weights
        self.weights = np.zeros(n_features)
        
        # Gradient Descent
        for i in range(self.n_iterations):
            # Prediction
            y_pred = X.dot(self.weights) + self.bias
            
            # Calculate errors
            error = y_pred - y
            
            # Calculate gradients
            dw = (1 / n_samples) * X.T.dot(error)
            db = (1 / n_samples) * np.sum(error)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Calculate cost (MSE)
            cost = np.mean(error ** 2)
            self.cost_history.append(cost)
            
            if (i + 1) % 100 == 0:
                print(f"Iteration {i + 1}/{self.n_iterations}, Cost: {cost:.4f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Dự đoán giá trị cho dữ liệu mới
        
        Args:
            X (np.ndarray): Feature matrix, shape (n_samples, n_features)
        
        Returns:
            np.ndarray: Predicted values
        """
        return X.dot(self.weights) + self.bias
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Tính R² score (Coefficient of Determination)
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target vector
        
        Returns:
            float: R² score (0 to 1, cao hơn là tốt hơn)
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def get_params(self) -> dict:
        """Trả về parameters của mô hình"""
        return {
            'weights': self.weights,
            'bias': self.bias,
            'learning_rate': self.learning_rate,
            'n_iterations': self.n_iterations
        }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Tạo dữ liệu mẫu
    np.random.seed(42)
    X = np.random.rand(100, 1) * 10  # 100 samples, 1 feature
    y = 2.5 * X.squeeze() + 1 + np.random.randn(100) * 2  # y = 2.5*X + 1 + noise
    
    # Chia dữ liệu train/test
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Tạo và huấn luyện mô hình
    model = LinearRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X_train, y_train)
    
    # Dự đoán
    y_pred = model.predict(X_test)
    
    # Đánh giá
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"\\nTrain R² Score: {train_score:.4f}")
    print(f"Test R² Score: {test_score:.4f}")
    print(f"Weights: {model.weights}")
    print(f"Bias: {model.bias}")
