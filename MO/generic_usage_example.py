# -*- coding: utf-8 -*-
"""
通用使用示例
展示如何将PredVAR模型应用到不同的数据集
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from core import PredVAR
from data_interface import DataLoader, DataPreprocessor, create_data_config

def auto_select_parameters(data: np.ndarray, max_l: int = 10, max_s: int = 20) -> tuple:
    """
    自动选择模型参数
    
    参数:
    - data: 输入数据 (N, p)
    - max_l: 最大潜变量维度
    - max_s: 最大VAR阶数
    
    返回:
    - l: 推荐的潜变量维度
    - s: 推荐的VAR阶数
    """
    N, p = data.shape
    
    # 基于数据维度选择潜变量维度
    if p <= 5:
        l = min(2, p-1)
    elif p <= 10:
        l = min(3, p-1)
    elif p <= 20:
        l = min(5, p-1)
    else:
        l = min(8, p-1)
    
    # 基于样本数选择VAR阶数
    if N <= 100:
        s = min(3, N//10)
    elif N <= 500:
        s = min(5, N//20)
    elif N <= 1000:
        s = min(8, N//30)
    else:
        s = min(12, N//50)
    
    # 确保参数合理
    l = max(1, min(l, max_l, p-1))
    s = max(1, min(s, max_s, N//10))
    
    print(f"自动选择参数: l={l}, s={s}")
    return l, s

def run_predvar_analysis(data: np.ndarray, 
                         feature_names: list,
                         config: dict = None) -> dict:
    """
    运行完整的PredVAR分析
    
    参数:
    - data: 输入数据 (N, p)
    - feature_names: 特征名称列表
    - config: 配置字典
    
    返回:
    - results: 分析结果字典
    """
    # 设置默认配置
    if config is None:
        config = create_data_config({})
    
    print("=" * 50)
    print("开始PredVAR分析")
    print("=" * 50)
    
    # 1. 数据预处理
    print("\n1. 数据预处理...")
    preprocessor = DataPreprocessor()
    
    # 处理缺失值
    if config['preprocessing']['handle_missing']:
        data = preprocessor.handle_missing_values(
            data, 
            method=config['preprocessing']['missing_method']
        )
    
    # 移除异常值
    if config['preprocessing']['remove_outliers']:
        data = preprocessor.remove_outliers(
            data, 
            method=config['preprocessing']['outlier_method']
        )
    
    # 标准化
    if config['preprocessing']['standardize']:
        data = preprocessor.standardize(
            data, 
            method=config['preprocessing']['standardize_method']
        )
    
    print(f"预处理后数据形状: {data.shape}")
    
    # 2. 自动选择参数
    print("\n2. 参数选择...")
    if config['model_params']['l'] is None or config['model_params']['s'] is None:
        l, s = auto_select_parameters(data)
        if config['model_params']['l'] is None:
            config['model_params']['l'] = l
        if config['model_params']['s'] is None:
            config['model_params']['s'] = s
    
    print(f"使用参数: l={config['model_params']['l']}, s={config['model_params']['s']}")
    
    # 3. 数据划分
    print("\n3. 数据划分...")
    N = data.shape[0]
    train_size = int(config['validation']['train_ratio'] * N)
    s = config['model_params']['s']
    
    y_train = data[:train_size, :]
    y_test = data[train_size - s:, :]  # 测试集需要包含s个历史样本
    
    print(f"训练集大小: {y_train.shape}")
    print(f"测试集大小: {y_test.shape}")
    
    # 4. 模型训练
    print("\n4. 模型训练...")
    model = PredVAR(
        l=config['model_params']['l'],
        s=config['model_params']['s'],
        max_iter=config['model_params']['max_iter'],
        tol=config['model_params']['tol'],
        verbose=config['model_params']['verbose']
    )
    
    model.fit(y_train)
    print("模型训练完成")
    
    # 5. 模型评估
    print("\n5. 模型评估...")
    metrics = model.score(y_test)
    print("测试集性能指标:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")
    
    # 6. 结果可视化
    print("\n6. 生成可视化结果...")
    
    # 绘制潜变量
    model.plot_dlv(y_train, title="Dynamic Latent Variables")
    
    # 绘制加载系数
    model.plot_loadings(feature_names=feature_names, title="DLV Loading Coefficients")
    
    # 绘制自相关函数
    model.plot_autocorrelation(y_train, max_lag=20, title="Autocorrelation Analysis")
    
    # 7. 预测示例
    print("\n7. 预测示例...")
    y_pred, v_pred = model.predict(y_test)
    
    # 绘制预测对比
    plt.figure(figsize=(12, 8))
    for i in range(min(3, data.shape[1])):  # 只显示前3个特征
        plt.subplot(3, 1, i + 1)
        plt.plot(y_test[s:s+100, i], label=f'True {feature_names[i]}', linewidth=2)
        plt.plot(y_pred[:100, i], label=f'Predicted {feature_names[i]}', linestyle='--', linewidth=2)
        plt.title(f'Prediction vs True Value - {feature_names[i]}')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 8. 返回结果
    results = {
        'model': model,
        'metrics': metrics,
        'config': config,
        'data_info': {
            'original_shape': data.shape,
            'train_shape': y_train.shape,
            'test_shape': y_test.shape,
            'feature_names': feature_names
        }
    }
    
    return results

def example_csv_analysis():
    """CSV数据文件分析示例"""
    print("CSV数据文件分析示例")
    print("=" * 50)
    
    # 创建示例CSV数据
    np.random.seed(42)
    n_samples = 1000
    n_features = 8
    
    # 生成具有时间相关性的数据
    t = np.linspace(0, 10, n_samples)
    data = np.zeros((n_samples, n_features))
    
    # 创建几个具有不同频率的正弦波
    for i in range(n_features):
        freq = 0.5 + i * 0.2
        phase = i * np.pi / 4
        data[:, i] = np.sin(2 * np.pi * freq * t + phase) + 0.1 * np.random.randn(n_samples)
    
    # 添加一些噪声
    data += 0.05 * np.random.randn(n_samples, n_features)
    
    # 创建DataFrame并保存为CSV
    feature_names = [f'Signal_{i+1}' for i in range(n_features)]
    df = pd.DataFrame(data, columns=feature_names)
    df.to_csv('example_data.csv', index=False)
    
    # 加载数据
    loader = DataLoader()
    data, feature_names = loader.load_csv('example_data.csv')
    
    # 配置
    config = create_data_config({
        'model_params': {
            'l': 3,  # 手动指定潜变量维度
            's': 5,  # 手动指定VAR阶数
            'max_iter': 50,
            'verbose': True
        },
        'preprocessing': {
            'remove_outliers': False,  # 对于仿真数据，不移除异常值
            'standardize': True
        }
    })
    
    # 运行分析
    results = run_predvar_analysis(data, feature_names, config)
    
    return results

def example_industrial_data():
    """工业数据示例（模拟）"""
    print("工业数据示例")
    print("=" * 50)
    
    # 生成模拟工业过程数据
    np.random.seed(123)
    n_samples = 2000
    n_features = 12
    
    # 创建具有多个时间尺度的工业过程数据
    t = np.linspace(0, 20, n_samples)
    data = np.zeros((n_samples, n_features))
    
    # 温度相关变量（慢变化）
    temp_base = 25 + 5 * np.sin(0.1 * t) + 0.5 * np.random.randn(n_samples)
    data[:, 0] = temp_base  # 主温度
    data[:, 1] = temp_base + 2 * np.random.randn(n_samples)  # 辅助温度
    data[:, 2] = temp_base - 1 + 0.5 * np.random.randn(n_samples)  # 环境温度
    
    # 压力相关变量（中等变化）
    pressure_base = 101.3 + 2 * np.sin(0.3 * t) + np.random.randn(n_samples)
    data[:, 3] = pressure_base  # 主压力
    data[:, 4] = pressure_base + 0.5 * np.random.randn(n_samples)  # 辅助压力
    
    # 流量相关变量（快变化）
    flow_base = 10 + 3 * np.sin(2 * t) + 0.3 * np.random.randn(n_samples)
    data[:, 5] = flow_base  # 主流量
    data[:, 6] = flow_base * 0.8 + 0.2 * np.random.randn(n_samples)  # 辅助流量
    
    # 其他传感器数据
    for i in range(7, n_features):
        data[:, i] = 50 + 10 * np.sin(0.5 * t + i) + 2 * np.random.randn(n_samples)
    
    # 特征名称
    feature_names = [
        'Temperature_Main', 'Temperature_Aux', 'Temperature_Env',
        'Pressure_Main', 'Pressure_Aux',
        'Flow_Main', 'Flow_Aux',
        'Sensor_1', 'Sensor_2', 'Sensor_3', 'Sensor_4', 'Sensor_5'
    ]
    
    # 配置
    config = create_data_config({
        'model_params': {
            'l': None,  # 自动选择
            's': None,  # 自动选择
            'max_iter': 100,
            'verbose': True
        },
        'preprocessing': {
            'remove_outliers': True,
            'outlier_method': 'iqr',
            'standardize': True
        }
    })
    
    # 运行分析
    results = run_predvar_analysis(data, feature_names, config)
    
    return results

def example_financial_data():
    """金融数据示例（模拟）"""
    print("金融数据示例")
    print("=" * 50)
    
    # 生成模拟金融时间序列数据
    np.random.seed(456)
    n_samples = 1500
    n_features = 10
    
    # 创建具有趋势和周期性的金融数据
    t = np.linspace(0, 5, n_samples)
    data = np.zeros((n_samples, n_features))
    
    # 股票价格（随机游走 + 趋势）
    for i in range(5):  # 5只股票
        trend = 0.001 * t + 0.1 * np.sin(0.5 * t)
        noise = np.cumsum(0.02 * np.random.randn(n_samples))
        data[:, i] = 100 + trend + noise
    
    # 技术指标
    for i in range(5, n_features):
        # 基于前5个股票价格计算技术指标
        base_price = np.mean(data[:, :5], axis=1)
        if i == 5:  # 移动平均
            window = 20
            data[:, i] = pd.Series(base_price).rolling(window=window).mean().fillna(base_price)
        elif i == 6:  # RSI
            delta = np.diff(base_price, prepend=base_price[0])
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            rs = pd.Series(gain).rolling(window=14).mean() / pd.Series(loss).rolling(window=14).mean()
            data[:, i] = 100 - (100 / (1 + rs.fillna(50)))
        else:  # 其他指标
            data[:, i] = base_price + 2 * np.random.randn(n_samples)
    
    # 特征名称
    feature_names = [
        'Stock_1', 'Stock_2', 'Stock_3', 'Stock_4', 'Stock_5',
        'MA_20', 'RSI', 'MACD', 'Volume', 'Volatility'
    ]
    
    # 配置
    config = create_data_config({
        'model_params': {
            'l': 4,  # 金融数据通常有4-5个主要因子
            's': 8,  # 金融数据需要较长的历史
            'max_iter': 150,
            'verbose': True
        },
        'preprocessing': {
            'remove_outliers': True,
            'outlier_method': 'zscore',
            'standardize': True
        }
    })
    
    # 运行分析
    results = run_predvar_analysis(data, feature_names, config)
    
    return results

def main():
    """主函数：运行所有示例"""
    print("PredVAR通用使用示例")
    print("=" * 60)
    
    # 示例1：CSV数据文件
    try:
        results1 = example_csv_analysis()
        print("\nCSV数据示例完成")
    except Exception as e:
        print(f"CSV数据示例失败: {e}")
    
    # 示例2：工业数据
    try:
        results2 = example_industrial_data()
        print("\n工业数据示例完成")
    except Exception as e:
        print(f"工业数据示例失败: {e}")
    
    # 示例3：金融数据
    try:
        results3 = example_financial_data()
        print("\n金融数据示例完成")
    except Exception as e:
        print(f"金融数据示例失败: {e}")
    
    print("\n所有示例运行完成！")

if __name__ == "__main__":
    main()

