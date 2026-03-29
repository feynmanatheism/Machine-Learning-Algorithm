"""
Metrics and Evaluation Functions
Module: utils/metrics.py

Description:
    Các hàm tính toán metrics để đánh giá mô hình
"""

import numpy as np
from typing import Union, Tuple


# ============================================================================
# CLASSIFICATION METRICS
# ============================================================================

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Độ chính xác (Accuracy)
    
    Công thức: (TP + TN) / (TP + TN + FP + FN)
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        float: Accuracy score (0 to 1)
    """
    return np.mean(y_true == y_pred)


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Độ chính xác dương tính (Precision)
    
    Công thức: TP / (TP + FP)
    Nghĩa: Trong những cái được dự đoán là dương, bao nhiêu thực sự đúng
    
    Args:
        y_true: True labels (binary)
        y_pred: Predicted labels (binary)
    
    Returns:
        float: Precision score
    """
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    
    if tp + fp == 0:
        return 0.0
    
    return tp / (tp + fp)


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Độ phủ (Recall / Sensitivity / True Positive Rate)
    
    Công thức: TP / (TP + FN)
    Nghĩa: Trong những cái thực sự là dương, bao nhiêu được phát hiện
    
    Args:
        y_true: True labels (binary)
        y_pred: Predicted labels (binary)
    
    Returns:
        float: Recall score
    """
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    if tp + fn == 0:
        return 0.0
    
    return tp / (tp + fn)


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    F1 Score (Điều hòa của Precision và Recall)
    
    Công thức: 2 * (Precision * Recall) / (Precision + Recall)
    
    Args:
        y_true: True labels (binary)
        y_pred: Predicted labels (binary)
    
    Returns:
        float: F1 score
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    
    if p + r == 0:
        return 0.0
    
    return 2 * (p * r) / (p + r)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Confusion Matrix
    
    Returns:
        np.ndarray: 2x2 confusion matrix [[TN, FP], [FN, TP]]
    """
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tp = np.sum((y_pred == 1) & (y_true == 1))
    
    return np.array([[tn, fp], [fn, tp]])


# ============================================================================
# REGRESSION METRICS
# ============================================================================

def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Squared Error (MSE)
    
    Công thức: (1/n) * Σ(y_true - y_pred)²
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        float: MSE (nhỏ hơn là tốt hơn)
    """
    return np.mean((y_true - y_pred) ** 2)


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error (MAE)
    
    Công thức: (1/n) * Σ|y_true - y_pred|
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        float: MAE (nhỏ hơn là tốt hơn)
    """
    return np.mean(np.abs(y_true - y_pred))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error (RMSE)
    
    Công thức: √(MSE)
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        float: RMSE
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R² Score (Coefficient of Determination)
    
    Công thức: 1 - (SS_res / SS_tot)
    Ý nghĩa: Bao nhiêu % của variance được giải thích bởi mô hình
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        float: R² score (0 to 1, cao hơn là tốt hơn)
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return 0.0
    
    return 1 - (ss_res / ss_tot)


# ============================================================================
# CLUSTERING METRICS
# ============================================================================

def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Silhouette Score
    
    Đánh giá độ tốt của phân cụm (-1 to 1, cao hơn là tốt hơn)
    
    Args:
        X: Feature matrix
        labels: Cluster labels
    
    Returns:
        float: Average silhouette score
    """
    from scipy.spatial.distance import cdist
    
    n_samples = X.shape[0]
    unique_labels = np.unique(labels)
    
    silhouette_vals = []
    
    for label in unique_labels:
        cluster_mask = labels == label
        cluster_points = X[cluster_mask]
        
        for point in cluster_points:
            # Intra-cluster distance
            a = np.mean(np.linalg.norm(cluster_points - point, axis=1))
            
            # Inter-cluster distance
            b_values = []
            for other_label in unique_labels:
                if other_label != label:
                    other_points = X[labels == other_label]
                    b = np.mean(np.linalg.norm(other_points - point, axis=1))
                    b_values.append(b)
            
            b = min(b_values) if b_values else a
            
            # Silhouette coefficient
            if max(a, b) != 0:
                s = (b - a) / max(a, b)
            else:
                s = 0
            
            silhouette_vals.append(s)
    
    return np.mean(silhouette_vals) if silhouette_vals else 0.0


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Classification example
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 1])
    
    print("Classification Metrics:")
    print(f"Accuracy: {accuracy(y_true, y_pred):.4f}")
    print(f"Precision: {precision(y_true, y_pred):.4f}")
    print(f"Recall: {recall(y_true, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_true, y_pred):.4f}")
    print(f"Confusion Matrix:\\n{confusion_matrix(y_true, y_pred)}")
    
    # Regression example
    print("\\nRegression Metrics:")
    y_true_reg = np.array([3, -0.5, 2, 7])
    y_pred_reg = np.array([2.5, 0.0, 2, 8])
    
    print(f"MSE: {mean_squared_error(y_true_reg, y_pred_reg):.4f}")
    print(f"MAE: {mean_absolute_error(y_true_reg, y_pred_reg):.4f}")
    print(f"R² Score: {r_squared(y_true_reg, y_pred_reg):.4f}")
