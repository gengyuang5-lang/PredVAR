# 生成Lorenz仿真数据
y, v_true, _, _ = generate_lorenz_data(n_samples=10000, p=15, l=3, sigma=0.1)

# 划分训练集和测试集
s = 12  # VAR阶数
train_size = 7000
y_train = y[:train_size, :]
y_test = y[train_size - s:, :]

# 训练模型
model = PredVAR(l=3, s=12, max_iter=50, verbose=True)
model.fit(y_train)

# 评估模型
metrics = model.score(y_test)
print("模型性能指标:")
print(f"RMSE: {metrics['rmse']:.6f}")
print(f"R²: {metrics['r2']:.6f}")
print(f"潜变量相似度: {metrics['similarity_v']:.6f}")

# 可视化
model.plot_dlv(y_train)
model.plot_loadings()
model.plot_autocorrelation(y_train)