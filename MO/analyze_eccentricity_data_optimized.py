# -*- coding: utf-8 -*-
"""
偏心故障数据集分析 - 优化版本
处理大数据集的内存优化版本
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from core import PredVAR
from data_interface import DataLoader, DataPreprocessor, create_data_config

def analyze_eccentricity_data_optimized():
    """分析偏心故障数据集 - 优化版本"""
    print("=" * 60)
    print("偏心故障数据集分析 - 优化版本")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n1. 加载数据...")
    loader = DataLoader()
    
    try:
        data, feature_names = loader.load_csv(
            'dataset/eccentricity.csv/eccentricity.csv',
            target_columns=['sensor1', 'sensor2', 'speedSet', 'load_value'],
            time_column='time_x',
            encoding='utf-8'
        )
        print(f"成功加载数据，形状: {data.shape}")
        print(f"特征名称: {feature_names}")
        
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None
    
    # 2. 数据采样（减少数据量以节省内存）
    print("\n2. 数据采样...")
    original_size = data.shape[0]
    
    # 使用每10个样本取1个，减少到15,000个样本
    sample_rate = 10
    data_sampled = data[::sample_rate, :]
    print(f"原始数据: {original_size} 样本")
    print(f"采样后数据: {data_sampled.shape[0]} 样本 (采样率: 1/{sample_rate})")
    
    # 3. 数据预处理
    print("\n3. 数据预处理...")
    preprocessor = DataPreprocessor()
    
    # 检查缺失值
    if np.isnan(data_sampled).any():
        print("发现缺失值，进行处理...")
        data_sampled = preprocessor.handle_missing_values(data_sampled, method='forward_fill')
    
    # 标准化数据
    print("标准化数据...")
    data_standardized = preprocessor.standardize(data_sampled, method='zscore')
    
    # 4. 数据可视化（显示前1000个样本）
    print("\n4. 数据可视化...")
    n_show = min(1000, data_standardized.shape[0])
    time_axis = np.arange(n_show)
    
    plt.figure(figsize=(15, 10))
    for i, name in enumerate(feature_names):
        plt.subplot(2, 2, i + 1)
        plt.plot(time_axis, data_standardized[:n_show, i], linewidth=1, alpha=0.8)
        plt.title(f'{name} - 标准化数据 (前{n_show}个样本)')
        plt.xlabel('时间步')
        plt.ylabel('标准化数值')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 5. 参数选择（针对采样后的数据）
    print("\n5. 参数选择...")
    N, p = data_standardized.shape
    
    # 根据采样后的数据选择参数
    if p <= 4:
        l = 2
    else:
        l = min(3, p-1)
    
    # 根据样本数选择VAR阶数
    if N <= 1000:
        s = min(3, N//20)
    elif N <= 5000:
        s = min(5, N//30)
    else:
        s = min(8, N//50)
    
    print(f"选择参数: l={l}, s={s}")
    
    # 6. 配置模型
    config = create_data_config({
        'model_params': {
            'l': l,
            's': s,
            'max_iter': 50,  # 减少迭代次数
            'tol': 1e-6,
            'verbose': True
        },
        'preprocessing': {
            'remove_outliers': False,
            'handle_missing': False,
            'standardize': False
        },
        'validation': {
            'train_ratio': 0.8,
            'test_ratio': 0.2,
            'min_samples': 50
        }
    })
    
    # 7. 运行PredVAR分析
    print("\n6. 运行PredVAR分析...")
    try:
        # 数据划分
        N = data_standardized.shape[0]
        train_size = int(0.8 * N)
        y_train = data_standardized[:train_size, :]
        y_test = data_standardized[train_size - s:, :]
        
        print(f"训练集大小: {y_train.shape}")
        print(f"测试集大小: {y_test.shape}")
        
        # 训练模型
        print("训练模型...")
        model = PredVAR(
            l=l,
            s=s,
            max_iter=50,
            tol=1e-6,
            verbose=True
        )
        
        model.fit(y_train)
        print("模型训练完成")
        
        # 8. 模型评估
        print("\n7. 模型评估...")
        metrics = model.score(y_test)
        print("测试集性能指标:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.6f}")
        
        # 9. 可视化结果
        print("\n8. 生成可视化结果...")
        
        # 绘制潜变量
        model.plot_dlv(y_train, title="Dynamic Latent Variables (Sampled Data)")
        
        # 绘制加载系数
        model.plot_loadings(feature_names=feature_names, title="DLV Loading Coefficients")
        
        # 绘制自相关函数
        model.plot_autocorrelation(y_train, max_lag=20, title="Autocorrelation Analysis")
        
        # 10. 预测示例
        print("\n9. 预测示例...")
        y_pred, v_pred = model.predict(y_test)
        
        # 绘制预测对比（显示前200个样本）
        n_show_pred = min(200, y_test.shape[0] - s)
        plt.figure(figsize=(15, 8))
        
        for i in range(min(2, data_standardized.shape[1])):  # 只显示前2个特征
            plt.subplot(2, 1, i + 1)
            plt.plot(y_test[s:s+n_show_pred, i], label=f'True {feature_names[i]}', linewidth=2)
            plt.plot(y_pred[:n_show_pred, i], label=f'Predicted {feature_names[i]}', linestyle='--', linewidth=2)
            plt.title(f'Prediction vs True Value - {feature_names[i]} (前{n_show_pred}个样本)')
            plt.xlabel('Time Step')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 11. 故障检测分析
        print("\n10. 故障检测分析...")
        
        # 计算预测误差
        prediction_errors = np.linalg.norm(y_test[s:] - y_pred, axis=1)
        
        # 计算误差阈值
        error_threshold = np.mean(prediction_errors) + 3 * np.std(prediction_errors)
        anomaly_mask = prediction_errors > error_threshold
        anomaly_count = np.sum(anomaly_mask)
        
        print(f"预测误差统计:")
        print(f"  平均误差: {np.mean(prediction_errors):.6f}")
        print(f"  误差标准差: {np.std(prediction_errors):.6f}")
        print(f"  异常阈值: {error_threshold:.6f}")
        print(f"  检测到异常点: {anomaly_count} 个 ({anomaly_count/len(prediction_errors)*100:.2f}%)")
        
        # 12. 异常检测可视化
        plt.figure(figsize=(15, 8))
        
        # 预测误差时间序列
        plt.subplot(2, 1, 1)
        plt.plot(prediction_errors, linewidth=1, alpha=0.8, label='预测误差')
        plt.axhline(y=error_threshold, color='red', linestyle='--', label=f'异常阈值 ({error_threshold:.4f})')
        plt.title('预测误差时间序列')
        plt.xlabel('时间步')
        plt.ylabel('预测误差')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 异常点分布
        plt.subplot(2, 1, 2)
        plt.scatter(np.arange(len(prediction_errors)), prediction_errors, 
                   c=anomaly_mask, cmap='RdYlBu', alpha=0.6, s=1)
        plt.axhline(y=error_threshold, color='red', linestyle='--', label='异常阈值')
        plt.title('异常点检测结果')
        plt.xlabel('时间步')
        plt.ylabel('预测误差')
        plt.colorbar(label='异常点')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 13. 保存结果
        print("\n11. 保存分析结果...")
        
        # 创建结果DataFrame
        results_df = pd.DataFrame({
            'time_step': np.arange(len(prediction_errors)),
            'prediction_error': prediction_errors,
            'is_anomaly': anomaly_mask,
            'sensor1_original': y_test[s:, 0],
            'sensor1_predicted': y_pred[:, 0],
            'sensor2_original': y_test[s:, 1],
            'sensor2_predicted': y_pred[:, 1]
        })
        
        # 保存到CSV
        results_df.to_csv('eccentricity_analysis_results_optimized.csv', index=False)
        print("分析结果已保存到: eccentricity_analysis_results_optimized.csv")
        
        # 保存模型参数
        model_info = {
            'model_parameters': {
                'l': l,
                's': s,
                'max_iter': 50,
                'tol': 1e-6,
                'sample_rate': sample_rate
            },
            'performance_metrics': metrics,
            'data_info': {
                'original_samples': original_size,
                'sampled_samples': data_sampled.shape[0],
                'features': feature_names,
                'anomaly_count': anomaly_count,
                'anomaly_percentage': anomaly_count/len(prediction_errors)*100
            }
        }
        
        import json
        with open('eccentricity_model_info_optimized.json', 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)
        print("模型信息已保存到: eccentricity_model_info_optimized.json")
        
        print("\n" + "=" * 60)
        print("偏心故障数据集分析完成！")
        print("=" * 60)
        print(f"原始数据: {original_size} 样本")
        print(f"分析数据: {data_sampled.shape[0]} 样本 (采样率: 1/{sample_rate})")
        print(f"模型参数: l={l}, s={s}")
        print(f"异常检测: {anomaly_count} 个异常点 ({anomaly_count/len(prediction_errors)*100:.2f}%)")
        
        return {
            'model': model,
            'metrics': metrics,
            'config': config,
            'data_info': model_info['data_info']
        }
        
    except Exception as e:
        print(f"PredVAR分析失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """主函数"""
    try:
        results = analyze_eccentricity_data_optimized()
        if results:
            print("\n分析成功完成！")
        else:
            print("分析失败，请检查数据和配置。")
    except Exception as e:
        print(f"程序运行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

