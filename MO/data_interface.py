# -*- coding: utf-8 -*-
"""
通用数据接口模块
支持多种数据格式的加载和预处理
"""
import numpy as np
import pandas as pd
import os
from typing import Union, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    """
    通用数据加载器
    支持CSV、Excel、NumPy数组等多种数据格式
    """
    
    def __init__(self):
        self.data = None
        self.feature_names = None
        self.data_info = {}
    
    def load_csv(self, file_path: str, 
                 target_columns: Optional[List[str]] = None,
                 time_column: Optional[str] = None,
                 encoding: str = 'utf-8') -> Tuple[np.ndarray, List[str]]:
        """
        加载CSV文件
        
        参数:
        - file_path: CSV文件路径
        - target_columns: 目标列名列表，None表示使用所有数值列
        - time_column: 时间列名，如果指定则从数据中移除
        - encoding: 文件编码
        
        返回:
        - data: 数据数组 (N, p)
        - feature_names: 特征名称列表
        """
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"成功加载CSV文件: {file_path}")
            print(f"数据形状: {df.shape}")
            
            # 移除时间列（如果指定）
            if time_column and time_column in df.columns:
                df = df.drop(columns=[time_column])
                print(f"已移除时间列: {time_column}")
            
            # 选择目标列
            if target_columns:
                missing_cols = [col for col in target_columns if col not in df.columns]
                if missing_cols:
                    raise ValueError(f"以下列不存在: {missing_cols}")
                df = df[target_columns]
            else:
                # 自动选择数值列
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df = df[numeric_cols]
            
            # 检查缺失值
            if df.isnull().any().any():
                print("警告: 数据中存在缺失值，将使用前向填充处理")
                df = df.fillna(method='ffill').fillna(method='bfill')
            
            self.data = df.values
            self.feature_names = df.columns.tolist()
            self.data_info = {
                'file_path': file_path,
                'data_type': 'csv',
                'shape': df.shape,
                'columns': self.feature_names
            }
            
            return self.data, self.feature_names
            
        except Exception as e:
            raise ValueError(f"加载CSV文件失败: {str(e)}")
    
    def load_excel(self, file_path: str,
                   sheet_name: Union[str, int] = 0,
                   target_columns: Optional[List[str]] = None,
                   time_column: Optional[str] = None) -> Tuple[np.ndarray, List[str]]:
        """
        加载Excel文件
        
        参数:
        - file_path: Excel文件路径
        - sheet_name: 工作表名称或索引
        - target_columns: 目标列名列表
        - time_column: 时间列名
        
        返回:
        - data: 数据数组 (N, p)
        - feature_names: 特征名称列表
        """
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            print(f"成功加载Excel文件: {file_path}")
            print(f"数据形状: {df.shape}")
            
            # 移除时间列（如果指定）
            if time_column and time_column in df.columns:
                df = df.drop(columns=[time_column])
                print(f"已移除时间列: {time_column}")
            
            # 选择目标列
            if target_columns:
                missing_cols = [col for col in target_columns if col not in df.columns]
                if missing_cols:
                    raise ValueError(f"以下列不存在: {missing_cols}")
                df = df[target_columns]
            else:
                # 自动选择数值列
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df = df[numeric_cols]
            
            # 检查缺失值
            if df.isnull().any().any():
                print("警告: 数据中存在缺失值，将使用前向填充处理")
                df = df.fillna(method='ffill').fillna(method='bfill')
            
            self.data = df.values
            self.feature_names = df.columns.tolist()
            self.data_info = {
                'file_path': file_path,
                'data_type': 'excel',
                'sheet_name': sheet_name,
                'shape': df.shape,
                'columns': self.feature_names
            }
            
            return self.data, self.feature_names
            
        except Exception as e:
            raise ValueError(f"加载Excel文件失败: {str(e)}")
    
    def load_numpy(self, data: np.ndarray, 
                   feature_names: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
        """
        加载NumPy数组
        
        参数:
        - data: NumPy数组 (N, p)
        - feature_names: 特征名称列表
        
        返回:
        - data: 数据数组 (N, p)
        - feature_names: 特征名称列表
        """
        if not isinstance(data, np.ndarray):
            raise ValueError("输入必须是NumPy数组")
        
        if data.ndim != 2:
            raise ValueError("数据必须是二维数组 (N, p)")
        
        self.data = data
        if feature_names is None:
            self.feature_names = [f'Feature_{i+1}' for i in range(data.shape[1])]
        else:
            if len(feature_names) != data.shape[1]:
                raise ValueError(f"特征名称数量({len(feature_names)})与数据维度({data.shape[1]})不匹配")
            self.feature_names = feature_names
        
        self.data_info = {
            'data_type': 'numpy',
            'shape': data.shape,
            'columns': self.feature_names
        }
        
        return self.data, self.feature_names
    
    def get_data_info(self) -> dict:
        """获取数据信息"""
        return self.data_info.copy()
    
    def validate_data(self, min_samples: int = 10) -> bool:
        """
        验证数据质量
        
        参数:
        - min_samples: 最小样本数
        
        返回:
        - bool: 数据是否有效
        """
        if self.data is None:
            print("错误: 未加载数据")
            return False
        
        N, p = self.data.shape
        
        if N < min_samples:
            print(f"错误: 样本数({N})少于最小要求({min_samples})")
            return False
        
        if p < 2:
            print(f"错误: 特征数({p})少于最小要求(2)")
            return False
        
        # 检查数值范围
        if np.isnan(self.data).any():
            print("警告: 数据中存在NaN值")
        
        if np.isinf(self.data).any():
            print("警告: 数据中存在无穷大值")
        
        print(f"数据验证通过: 形状({N}, {p}), 特征: {self.feature_names}")
        return True


class DataPreprocessor:
    """
    数据预处理器
    提供数据清洗、标准化等功能
    """
    
    def __init__(self):
        self.scaler = None
        self.outlier_threshold = 3.0  # 异常值阈值（标准差倍数）
    
    def remove_outliers(self, data: np.ndarray, method: str = 'iqr') -> np.ndarray:
        """
        移除异常值
        
        参数:
        - data: 输入数据 (N, p)
        - method: 异常值检测方法 ('iqr', 'zscore')
        
        返回:
        - cleaned_data: 清洗后的数据
        """
        if method == 'iqr':
            # 使用四分位数间距方法
            Q1 = np.percentile(data, 25, axis=0)
            Q3 = np.percentile(data, 75, axis=0)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # 标记异常值
            outlier_mask = np.any((data < lower_bound) | (data > upper_bound), axis=1)
            
        elif method == 'zscore':
            # 使用Z-score方法
            z_scores = np.abs((data - np.mean(data, axis=0)) / np.std(data, axis=0))
            outlier_mask = np.any(z_scores > self.outlier_threshold, axis=1)
        
        else:
            raise ValueError(f"不支持的异常值检测方法: {method}")
        
        cleaned_data = data[~outlier_mask]
        removed_count = np.sum(outlier_mask)
        
        if removed_count > 0:
            print(f"移除了 {removed_count} 个异常值样本")
        
        return cleaned_data
    
    def standardize(self, data: np.ndarray, method: str = 'zscore') -> np.ndarray:
        """
        数据标准化
        
        参数:
        - data: 输入数据 (N, p)
        - method: 标准化方法 ('zscore', 'minmax', 'robust')
        
        返回:
        - standardized_data: 标准化后的数据
        """
        if method == 'zscore':
            # Z-score标准化
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            standardized_data = (data - mean) / (std + 1e-8)
            
        elif method == 'minmax':
            # Min-Max标准化
            min_val = np.min(data, axis=0)
            max_val = np.max(data, axis=0)
            standardized_data = (data - min_val) / (max_val - min_val + 1e-8)
            
        elif method == 'robust':
            # 鲁棒标准化（使用中位数和MAD）
            median = np.median(data, axis=0)
            mad = np.median(np.abs(data - median), axis=0)
            standardized_data = (data - median) / (mad + 1e-8)
            
        else:
            raise ValueError(f"不支持的标准化方法: {method}")
        
        return standardized_data
    
    def handle_missing_values(self, data: np.ndarray, method: str = 'forward_fill') -> np.ndarray:
        """
        处理缺失值
        
        参数:
        - data: 输入数据 (N, p)
        - method: 处理方法 ('forward_fill', 'backward_fill', 'interpolate', 'mean')
        
        返回:
        - cleaned_data: 处理后的数据
        """
        if not np.isnan(data).any():
            return data
        
        if method == 'forward_fill':
            # 前向填充
            df = pd.DataFrame(data)
            df = df.fillna(method='ffill')
            cleaned_data = df.values
            
        elif method == 'backward_fill':
            # 后向填充
            df = pd.DataFrame(data)
            df = df.fillna(method='bfill')
            cleaned_data = df.values
            
        elif method == 'interpolate':
            # 线性插值
            df = pd.DataFrame(data)
            df = df.interpolate()
            cleaned_data = df.values
            
        elif method == 'mean':
            # 均值填充
            mean_values = np.nanmean(data, axis=0)
            cleaned_data = data.copy()
            for i in range(data.shape[1]):
                mask = np.isnan(cleaned_data[:, i])
                cleaned_data[mask, i] = mean_values[i]
        
        else:
            raise ValueError(f"不支持的缺失值处理方法: {method}")
        
        return cleaned_data


def create_data_config(config_dict: dict) -> dict:
    """
    创建数据配置
    
    参数:
    - config_dict: 配置字典
    
    返回:
    - config: 完整配置
    """
    default_config = {
        'data_source': {
            'type': 'csv',  # 'csv', 'excel', 'numpy'
            'file_path': None,
            'target_columns': None,
            'time_column': None,
            'encoding': 'utf-8'
        },
        'preprocessing': {
            'remove_outliers': True,
            'outlier_method': 'iqr',
            'handle_missing': True,
            'missing_method': 'forward_fill',
            'standardize': True,
            'standardize_method': 'zscore'
        },
        'model_params': {
            'l': None,  # 动态潜变量维度，None表示自动选择
            's': None,  # VAR阶数，None表示自动选择
            'max_iter': 100,
            'tol': 1e-6,
            'verbose': True
        },
        'validation': {
            'train_ratio': 0.7,
            'test_ratio': 0.3,
            'min_samples': 50
        }
    }
    
    # 更新默认配置
    for key, value in config_dict.items():
        if key in default_config:
            if isinstance(value, dict):
                default_config[key].update(value)
            else:
                default_config[key] = value
        else:
            default_config[key] = value
    
    return default_config

