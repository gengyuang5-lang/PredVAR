# 加载工业数据（示例）
# 假设数据格式为 (N+s, p)，每行一个样本，每列一个变量
import pandas as pd
data = pd.read_csv('industrial_data.csv')
y = data.values

# 数据划分
s = 7  # 根据PRESS分析选择的阶数
l = 5  # 动态潜变量维度
train_size = int(0.7 * len(y))
y_train = y[:train_size, :]
y_test = y[train_size - s:, :]

# 训练模型
model = PredVAR(l=l, s=s, max_iter=50, verbose=True)
model.fit(y_train)

# 评估
metrics = model.score(y_test)
print(f"测试集相对误差: {metrics['prediction_error']:.6f}")

# 可视化潜变量和加载系数
feature_names = data.columns.tolist()
model.plot_dlv(y_train, title="Industrial Process DLVs")
model.plot_loadings(feature_names=feature_names)