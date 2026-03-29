"""
Data Preprocessing Utilities
Module: utils/data_preprocessing.py

Description:
    Các hàm tiện ích cho xử lý và chuẩn bị dữ liệu trước khi đưa vào mô hình.
"""

import numpy as np
from typing import Tuple, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


def load_data(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dữ liệu từ file CSV
    
    Args:
        filepath (str): Đường dẫn tới file CSV
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Features (X) và Target (y)
    """
    try:
        import pandas as pd
        df = pd.read_csv(filepath)
        
        # Giả sử cột cuối cùng là target
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        
        return X, y
    except Exception as e:
        print(f"Lỗi khi load dữ liệu: {e}")
        return None, None


def split_data(X: np.ndarray, y: np.ndarray, 
               test_size: float = 0.2, random_state: int = 42) -> Tuple:
    """
    Chia dữ liệu thành train/test set
    
    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target vector
        test_size (float): Tỷ lệ test (0.0-1.0)
        random_state (int): Seed
    
    Returns:
        Tuple: X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def standardize(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Chuẩn hóa dữ liệu (mean=0, std=1)
    
    Args:
        X_train (np.ndarray): Training features
        X_test (np.ndarray): Test features
    
    Returns:
        Tuple: Chuẩn hóa X_train và X_test
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def normalize(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Chuẩn hóa dữ liệu vào khoảng [0, 1]
    
    Args:
        X_train (np.ndarray): Training features
        X_test (np.ndarray): Test features
    
    Returns:
        Tuple: Chuẩn hóa X_train và X_test
    """
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def handle_missing_values(X: np.ndarray, strategy: str = 'mean') -> np.ndarray:
    """
    Xử lý các giá trị thiếu
    
    Args:
        X (np.ndarray): Feature matrix
        strategy (str): 'mean', 'median', hoặc 'drop'
    
    Returns:
        np.ndarray: Dữ liệu sau xử lý
    """
    if strategy == 'mean':
        col_mean = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_mean, inds[1])
    elif strategy == 'median':
        col_median = np.nanmedian(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_median, inds[1])
    
    return X


def remove_outliers(X: np.ndarray, threshold: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loại bỏ outliers dựa trên Z-score
    
    Args:
        X (np.ndarray): Feature matrix
        threshold (float): Z-score threshold
    
    Returns:
        Tuple: Filtered features và indices of kept samples
    """
    from scipy import stats
    z_scores = np.abs(stats.zscore(X, axis=0))
    mask = (z_scores < threshold).all(axis=1)
    return X[mask], np.where(mask)[0]


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Tạo dữ liệu mẫu
    np.random.seed(42)
    X = np.random.rand(100, 5) * 100
    y = np.random.rand(100)
    
    # Chia dữ liệu
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    
    # Chuẩn hóa
    X_train_std, X_test_std = standardize(X_train, X_test)
    print(f"\\nChuẩn hóa - Mean: {X_train_std.mean():.4f}, Std: {X_train_std.std():.4f}")
    
    # Chuẩn hóa vào [0,1]
    X_train_norm, X_test_norm = normalize(X_train, X_test)
    print(f"Chuẩn hóa [0,1] - Min: {X_train_norm.min():.4f}, Max: {X_train_norm.max():.4f}")
