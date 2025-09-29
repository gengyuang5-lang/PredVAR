# PredVAR 通用使用指南

## 项目概述

这个项目实现了概率降维向量自回归模型（PredVAR），用于动态降维和去噪声。现在支持多种数据格式的通用接口。

## 快速开始

### 1. 基本使用（CSV文件）

```python
from core import PredVAR
from data_interface import DataLoader, create_data_config

# 加载数据
loader = DataLoader()
data, feature_names = loader.load_csv('your_data.csv')

# 创建配置
config = create_data_config({
    'model_params': {
        'l': 3,  # 潜变量维度
        's': 5,  # VAR阶数
        'max_iter': 100,
        'verbose': True
    }
})

# 运行分析
from generic_usage_example import run_predvar_analysis
results = run_predvar_analysis(data, feature_names, config)
```

### 2. 自动参数选择

```python
# 让系统自动选择参数
config = create_data_config({
    'model_params': {
        'l': None,  # 自动选择潜变量维度
        's': None,  # 自动选择VAR阶数
    }
})
```

### 3. 不同数据格式支持

#### CSV文件
```python
data, feature_names = loader.load_csv(
    'data.csv',
    target_columns=['col1', 'col2', 'col3'],  # 指定列
    time_column='timestamp'  # 移除时间列
)
```

#### Excel文件
```python
data, feature_names = loader.load_excel(
    'data.xlsx',
    sheet_name=0,  # 工作表索引或名称
    target_columns=['col1', 'col2', 'col3']
)
```

#### NumPy数组
```python
import numpy as np
data = np.random.randn(1000, 10)
feature_names = [f'Feature_{i}' for i in range(10)]

data, feature_names = loader.load_numpy(data, feature_names)
```

## 配置选项

### 数据预处理配置
```python
config = create_data_config({
    'preprocessing': {
        'remove_outliers': True,      # 是否移除异常值
        'outlier_method': 'iqr',      # 异常值检测方法 ('iqr', 'zscore')
        'handle_missing': True,       # 是否处理缺失值
        'missing_method': 'forward_fill',  # 缺失值处理方法
        'standardize': True,          # 是否标准化
        'standardize_method': 'zscore'  # 标准化方法
    }
})
```

### 模型参数配置
```python
config = create_data_config({
    'model_params': {
        'l': 3,           # 动态潜变量维度
        's': 5,           # VAR模型阶数
        'max_iter': 100,  # 最大迭代次数
        'tol': 1e-6,      # 收敛阈值
        'verbose': True   # 是否显示详细信息
    }
})
```

### 数据划分配置
```python
config = create_data_config({
    'validation': {
        'train_ratio': 0.7,    # 训练集比例
        'test_ratio': 0.3,      # 测试集比例
        'min_samples': 50       # 最小样本数
    }
})
```

## 参数选择建议

### 潜变量维度 (l)
- **小数据集** (p ≤ 5): l = 2
- **中等数据集** (5 < p ≤ 10): l = 3-4
- **大数据集** (p > 10): l = 5-8
- **自动选择**: 设置为 `None`

### VAR阶数 (s)
- **小样本** (N ≤ 100): s = 2-3
- **中等样本** (100 < N ≤ 500): s = 3-5
- **大样本** (N > 500): s = 5-12
- **自动选择**: 设置为 `None`

## 结果解释

### 性能指标
- **RMSE**: 均方根误差，越小越好
- **R²**: 决定系数，越接近1越好
- **similarity_v**: 潜变量相似度
- **similarity_y**: 测量值相似度

### 可视化结果
1. **动态潜变量图**: 显示提取的潜变量时间序列
2. **加载系数图**: 显示各特征对潜变量的贡献
3. **自相关图**: 分析潜变量和残差的时间相关性

## 常见问题

### Q: 如何选择合适的参数？
A: 建议先使用自动参数选择，然后根据结果调整。通常从较小的l和s开始尝试。

### Q: 数据需要预处理吗？
A: 建议进行标准化，对于有异常值的数据建议移除异常值。

### Q: 如何解释加载系数？
A: 加载系数表示各特征对潜变量的贡献程度，绝对值越大表示贡献越大。

### Q: 模型不收敛怎么办？
A: 可以尝试增加最大迭代次数或调整收敛阈值。

## 完整示例

```python
# 1. 导入必要模块
from core import PredVAR
from data_interface import DataLoader, create_data_config
from generic_usage_example import run_predvar_analysis

# 2. 加载数据
loader = DataLoader()
data, feature_names = loader.load_csv('your_data.csv')

# 3. 创建配置
config = create_data_config({
    'model_params': {
        'l': None,  # 自动选择
        's': None,  # 自动选择
        'max_iter': 100,
        'verbose': True
    },
    'preprocessing': {
        'remove_outliers': True,
        'standardize': True
    }
})

# 4. 运行分析
results = run_predvar_analysis(data, feature_names, config)

# 5. 查看结果
print("模型性能:", results['metrics'])
print("数据信息:", results['data_info'])
```

## 注意事项

1. **数据格式**: 确保数据为数值型，无缺失值或已处理缺失值
2. **时间序列**: 数据应按时间顺序排列
3. **样本数量**: 建议至少50个样本
4. **特征数量**: 建议至少2个特征
5. **内存使用**: 大数据集可能需要较多内存

