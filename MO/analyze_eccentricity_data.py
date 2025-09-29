# -*- coding: utf-8 -*-
"""
偏心故障数据集分析
使用PredVAR模型进行动态降维和去噪声分析
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from core import PredVAR
from data_interface import DataLoader, DataPreprocessor, create_data_config
from generic_usage_example import run_predvar_analysis, auto_select_parameters

def analyze_eccentricity_data():
    """分析偏心故障数据集"""
    print("=" * 60)
    print("偏心故障数据集分析")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n1. 加载数据...")
    loader = DataLoader()
    
    try:
        data, feature_names = loader.load_csv(
            'dataset/eccentricity.csv/eccentricity.csv',
            target_columns=['sensor1', 'sensor2', 'speedSet', 'load_value'],  # 选择数值列
            time_column='time_x',  # 移除时间列
            encoding='utf-8'
        )
        print(f"成功加载数据，形状: {data.shape}")
        print(f"特征名称: {feature_names}")
        
        # 显示数据基本信息
        print(f"\n数据统计信息:")
        print(f"样本数: {data.shape[0]}")
        print(f"特征数: {data.shape[1]}")
        print(f"数据范围: [{np.min(data):.4f}, {np.max(data):.4f}]")
        print(f"数据均值: {np.mean(data, axis=0)}")
        print(f"数据标准差: {np.std(data, axis=0)}")
        
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None
    
    # 2. 数据预处理
    print("\n2. 数据预处理...")
    preprocessor = DataPreprocessor()
    
    # 检查缺失值
    if np.isnan(data).any():
        print("发现缺失值，进行处理...")
        data = preprocessor.handle_missing_values(data, method='forward_fill')
    
    # 检查异常值
    print("检查异常值...")
    Q1 = np.percentile(data, 25, axis=0)
    Q3 = np.percentile(data, 75, axis=0)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outlier_mask = np.any((data < lower_bound) | (data > upper_bound), axis=1)
    outlier_count = np.sum(outlier_mask)
    print(f"发现 {outlier_count} 个异常值样本 ({outlier_count/data.shape[0]*100:.2f}%)")
    
    # 标准化数据
    print("标准化数据...")
    data_standardized = preprocessor.standardize(data, method='zscore')
    
    # 3. 数据可视化
    print("\n3. 数据可视化...")
    
    # 原始数据时间序列图
    plt.figure(figsize=(15, 10))
    
    # 显示前1000个样本
    n_show = min(1000, data.shape[0])
    time_axis = np.arange(n_show)
    
    for i, name in enumerate(feature_names):
        plt.subplot(2, 2, i + 1)
        plt.plot(time_axis, data[:n_show, i], linewidth=1, alpha=0.8)
        plt.title(f'{name} - 原始数据')
        plt.xlabel('时间步')
        plt.ylabel('数值')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 标准化数据时间序列图
    plt.figure(figsize=(15, 10))
    
    for i, name in enumerate(feature_names):
        plt.subplot(2, 2, i + 1)
        plt.plot(time_axis, data_standardized[:n_show, i], linewidth=1, alpha=0.8)
        plt.title(f'{name} - 标准化数据')
        plt.xlabel('时间步')
        plt.ylabel('标准化数值')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 4. 自动参数选择
    print("\n4. 参数选择...")
    l, s = auto_select_parameters(data_standardized)
    print(f"自动选择参数: l={l}, s={s}")
    
    # 5. 配置模型
    config = create_data_config({
        'model_params': {
            'l': l,
            's': s,
            'max_iter': 100,
            'tol': 1e-6,
            'verbose': True
        },
        'preprocessing': {
            'remove_outliers': False,  # 对于传感器数据，保留所有数据
            'handle_missing': False,   # 已经处理过
            'standardize': False       # 已经标准化
        },
        'validation': {
            'train_ratio': 0.8,  # 使用80%数据训练
            'test_ratio': 0.2,   # 20%数据测试
            'min_samples': 100
        }
    })
    
    # 6. 运行PredVAR分析
    print("\n5. 运行PredVAR分析...")
    try:
        results = run_predvar_analysis(data_standardized, feature_names, config)
        
        # 7. 详细结果分析
        print("\n6. 详细结果分析...")
        model = results['model']
        metrics = results['metrics']
        
        print(f"\n模型性能指标:")
        print(f"  RMSE: {metrics['rmse']:.6f}")
        print(f"  R²: {metrics['r2']:.6f}")
        print(f"  潜变量相似度: {metrics['similarity_v']:.6f}")
        print(f"  测量值相似度: {metrics['similarity_y']:.6f}")
        print(f"  预测误差: {metrics['prediction_error']:.6f}")
        
        # 8. 潜变量分析
        print("\n7. 潜变量分析...")
        
        # 计算训练数据的潜变量
        y_train_centered = model.scaler.transform(data_standardized)
        v_train = model.R.T @ y_train_centered.T
        v_train = v_train.T
        
        print(f"潜变量统计信息:")
        for i in range(l):
            print(f"  DLV {i+1}: 均值={np.mean(v_train[:, i]):.4f}, 标准差={np.std(v_train[:, i]):.4f}")
        
        # 9. 故障检测分析
        print("\n8. 故障检测分析...")
        
        # 计算预测误差
        y_pred, v_pred = model.predict(data_standardized)
        prediction_errors = np.linalg.norm(data_standardized[s:] - y_pred, axis=1)
        
        # 计算误差阈值（基于3倍标准差）
        error_threshold = np.mean(prediction_errors) + 3 * np.std(prediction_errors)
        anomaly_mask = prediction_errors > error_threshold
        anomaly_count = np.sum(anomaly_mask)
        
        print(f"预测误差统计:")
        print(f"  平均误差: {np.mean(prediction_errors):.6f}")
        print(f"  误差标准差: {np.std(prediction_errors):.6f}")
        print(f"  异常阈值: {error_threshold:.6f}")
        print(f"  检测到异常点: {anomaly_count} 个 ({anomaly_count/len(prediction_errors)*100:.2f}%)")
        
        # 10. 异常检测可视化
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
        
        # 11. 保存结果
        print("\n9. 保存分析结果...")
        
        # 创建结果DataFrame
        results_df = pd.DataFrame({
            'time_step': np.arange(len(prediction_errors)),
            'prediction_error': prediction_errors,
            'is_anomaly': anomaly_mask,
            'sensor1_original': data[s:, 0],
            'sensor1_predicted': y_pred[:, 0],
            'sensor2_original': data[s:, 1],
            'sensor2_predicted': y_pred[:, 1]
        })
        
        # 保存到CSV
        results_df.to_csv('eccentricity_analysis_results.csv', index=False)
        print("分析结果已保存到: eccentricity_analysis_results.csv")
        
        # 保存模型参数
        model_info = {
            'model_parameters': {
                'l': l,
                's': s,
                'max_iter': config['model_params']['max_iter'],
                'tol': config['model_params']['tol']
            },
            'performance_metrics': metrics,
            'data_info': {
                'total_samples': data.shape[0],
                'features': feature_names,
                'anomaly_count': anomaly_count,
                'anomaly_percentage': anomaly_count/len(prediction_errors)*100
            }
        }
        
        import json
        with open('eccentricity_model_info.json', 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)
        print("模型信息已保存到: eccentricity_model_info.json")
        
        return results
        
    except Exception as e:
        print(f"PredVAR分析失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """主函数"""
    try:
        results = analyze_eccentricity_data()
        if results:
            print("\n" + "=" * 60)
            print("偏心故障数据集分析完成！")
            print("=" * 60)
        else:
            print("分析失败，请检查数据和配置。")
    except Exception as e:
        print(f"程序运行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

