# -*- coding: utf-8 -*-
import sys
import io
import os
# 设置标准输出和标准错误输出为UTF-8编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 在Windows PowerShell环境中尝试设置控制台编码
if os.name == 'nt':
    try:
        import ctypes
        # 设置Windows控制台为UTF-8编码
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleOutputCP(65001)  # UTF-8
        kernel32.SetConsoleCP(65001)  # UTF-8
    except:
        pass

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig, svd, inv, pinv
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 设置Matplotlib支持中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

class PredVAR:
    """
    概率降维向量自回归模型（PredVAR）实现
    
    参数:
    - l: 动态潜变量维度（DLD）
    - s: VAR模型阶数
    - max_iter: 最大迭代次数
    - tol: 收敛阈值
    - verbose: 打印迭代信息
    """
    def __init__(self, l, s, max_iter=100, tol=1e-6, verbose=False):
        self.l = l  # 动态潜变量维度
        self.s = s  # VAR模型阶数
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        
        # 模型参数
        self.B = None  # VAR系数矩阵 (s, l, l)
        self.P = None  # DLV加载矩阵 (p, l)
        self.R = None  # DLV权重矩阵 (p, l)
        self.P_bar = None  # 静态噪声加载矩阵 (p, p-l)
        self.R_bar = None  # 静态噪声权重矩阵 (p, p-l)
        self.Sigma_e = None  # 创新项协方差矩阵 (p, p)
        self.Sigma_eps_bar = None  # 静态噪声协方差矩阵 (p-l, p-l)
        self.Sigma_eps = None  # DLV创新项协方差矩阵 (l, l)
        
        # 预处理参数
        self.U = None  # 特征向量矩阵
        self.D = None  # 特征值矩阵
        self.r = None  # 有效数据维度
        self.scaler = None  # 标准化器
        
    def _preprocess(self, y):
        """
        数据预处理：中心化和特征值分解标准化
        
        参数:
        - y: 输入数据 (N+s, p)，N为样本数，p为特征数
        
        返回:
        - y_centered: 中心化后的数据 (N+s, p)
        - y_star: 标准化后的数据 (N+s, r)
        """
        # 中心化
        self.scaler = StandardScaler(with_std=False)
        y_centered = self.scaler.fit_transform(y)
        
        # 计算协方差矩阵并进行特征值分解
        N_total, p = y_centered.shape
        Sigma_y = np.cov(y_centered.T)  # (p, p)
        
        # 特征值分解（按降序排列）
        eig_vals, eig_vecs = eig(Sigma_y)
        eig_vals = np.real(eig_vals)
        self.eig_vecs = np.real(eig_vecs)
        
        # 排序
        self.idx = np.argsort(eig_vals)[::-1]
        eig_vals = eig_vals[self.idx]
        self.eig_vecs = self.eig_vecs[:, self.idx]
        
        # 去除零特征值对应的维度
        self.r = np.sum(eig_vals > 1e-10)  # 有效维度
        self.U = self.eig_vecs[:, :self.r]  # (p, r)
        self.D = np.diag(eig_vals[:self.r])  # (r, r)
        
        # 计算标准化数据 y* = D^(-1/2) U^T y
        D_sqrt_inv = inv(np.sqrt(self.D))  # (r, r)
        y_star = y_centered @ self.U @ D_sqrt_inv  # (N+s, r)
        
        return y_centered, y_star
    
    def _create_data_matrices(self, y_star):
        """
        创建数据矩阵 Y_i 和 V_i
        
        参数:
        - y_star: 标准化数据 (N+s, r)
        
        返回:
        - Y_list: Y_i 列表，每个元素 (N, r)，i=0到s
        - N: 有效样本数
        """
        N_total, r = y_star.shape
        N = N_total - self.s  # 有效样本数
        
        Y_list = []
        for i in range(self.s + 1):
            # Y_i: 第i个延迟的数据矩阵 (N, r)
            Y_i = y_star[i:i+N, :]
            Y_list.append(Y_i)
        
        return Y_list, N
    
    def _initialize_P_star(self, Y_list):
        """
        初始化 P* 矩阵（基于工具变量方法）
        
        参数:
        - Y_list: Y_i 列表
        
        返回:
        - P_star_init: 初始 P* 矩阵 (r, l)
        """
        Y_s = Y_list[self.s]  # (N, r)
        Y_others = np.hstack(Y_list[:self.s])  # (N, s*r)
        
        # 计算投影矩阵 Π_Y*
        Y_others_T = Y_others.T
        cov_Y = Y_others_T @ Y_others / len(Y_others)  # (s*r, s*r)
        cov_Y_inv = pinv(cov_Y)  # 伪逆
        Pi_Y = Y_others @ cov_Y_inv @ Y_others_T  # (N, N)
        
        # 计算 Y_s^T Π_Y* Y_s / N
        term = Y_s.T @ Pi_Y @ Y_s / len(Y_s)  # (r, r)
        
        # 特征值分解
        eig_vals, eig_vecs = eig(term)
        eig_vals = np.real(eig_vals)
        eig_vecs = np.real(eig_vecs)
        
        # 选择前l个特征向量
        idx = np.argsort(eig_vals)[::-1]
        P_star_init = eig_vecs[:, idx[:self.l]]  # (r, l)
        
        return P_star_init
    
    def _update_parameters(self, Y_list, P_star, N):
        """
        更新模型参数
        
        参数:
        - Y_list: Y_i 列表
        - P_star: 当前 P* 矩阵 (r, l)
        - N: 样本数
        
        返回:
        - B_new: 新的VAR系数 (s, l, l)
        - P_star_new: 新的 P* 矩阵 (r, l)
        - Sigma_hat_v: 预测方差矩阵 (l, l)
        """
        Y_s = Y_list[self.s]  # (N, r)
        V_list = []
        
        # 计算 V_i = Y_i P*
        for i in range(self.s + 1):
            V_i = Y_list[i] @ P_star  # (N, l)
            V_list.append(V_i)
        
        # 构建 V 矩阵 (N, s*l)
        V = np.hstack(V_list[:self.s])  # (N, s*l)
        V_s = V_list[self.s]  # (N, l)
        
        # 估计 VAR 系数 B
        V_T = V.T
        cov_V = V_T @ V / N  # (s*l, s*l)
        cov_V_inv = pinv(cov_V)
        B_flat = cov_V_inv @ V_T @ V_s / N  # (s*l, l)
        
        # 重塑为 (s, l, l)
        B_new = B_flat.reshape(self.s, self.l, self.l)
        
        # 计算预测值 V_hat_s
        V_hat_s = V @ B_flat  # (N, l)
        
        # 计算投影矩阵 Π_V_hat
        V_hat_s_T = V_hat_s.T
        cov_V_hat = V_hat_s_T @ V_hat_s / N  # (l, l)
        cov_V_hat_inv = pinv(cov_V_hat)
        Pi_V_hat = V_hat_s @ cov_V_hat_inv @ V_hat_s_T  # (N, N)
        
        # 计算 Y_s^T Π_V_hat Y_s / N 并进行特征值分解
        term = Y_s.T @ Pi_V_hat @ Y_s / N  # (r, r)
        eig_vals, eig_vecs = eig(term)
        eig_vals = np.real(eig_vals)
        eig_vecs = np.real(eig_vecs)
        
        # 选择前l个特征向量作为新的 P*
        idx = np.argsort(eig_vals)[::-1]
        P_star_new = eig_vecs[:, idx[:self.l]]  # (r, l)
        
        # 计算 Σ_hat_v
        Sigma_hat_v = V_hat_s_T @ V_hat_s / N  # (l, l)
        
        return B_new, P_star_new, Sigma_hat_v
    
    def fit(self, y):
        """
        模型训练
        
        参数:
        - y: 输入数据 (N+s, p)，N为样本数，p为特征数
        """
        N_total, p = y.shape
        if N_total < self.s + 1:
            raise ValueError(f"样本数必须大于VAR阶数s。当前: {N_total}, 需要至少: {self.s + 1}")
        
        # 1. 数据预处理
        y_centered, y_star = self._preprocess(y)
        
        # 2. 创建数据矩阵
        Y_list, N = self._create_data_matrices(y_star)
        
        # 3. 初始化 P*
        P_star = self._initialize_P_star(Y_list)
        
        # 4. 迭代优化
        prev_trace = 0
        for iter_idx in range(self.max_iter):
            # 更新参数
            B_new, P_star_new, Sigma_hat_v = self._update_parameters(Y_list, P_star, N)
            
            # 计算迹（用于收敛判断）
            current_trace = np.trace(Sigma_hat_v)
            
            # 检查收敛
            if abs(current_trace - prev_trace) < self.tol:
                if self.verbose:
                    print(f"迭代 {iter_idx+1} 收敛，迹变化: {abs(current_trace - prev_trace):.6f}")
                break
            
            # 更新参数
            self.B = B_new
            P_star = P_star_new
            prev_trace = current_trace
            
            if self.verbose and (iter_idx + 1) % 10 == 0:
                print(f"迭代 {iter_idx+1}, 迹: {current_trace:.6f}")
        
        if iter_idx == self.max_iter - 1 and self.verbose:
            print(f"达到最大迭代次数 {self.max_iter}，未完全收敛")
        
        # 5. 计算最终参数
        # 计算 P = U D^(1/2) P*
        D_sqrt = np.sqrt(self.D)  # (r, r)
        self.P = self.U @ D_sqrt @ P_star  # (p, l)
        
        # 计算 R = U D^(-1/2) P*
        D_sqrt_inv = inv(D_sqrt)  # (r, r)
        self.R = self.U @ D_sqrt_inv @ P_star  # (p, l)
        
        # 计算 P_bar 和 R_bar
        # P_bar = [U D^(1/2) P_bar* | U_tilde]
        # 其中 P_bar* 是与 P* 正交的向量
        if self.r > self.l:
            # 计算 P_bar* (与 P* 正交)
            P_bar_star = self.eig_vecs[:, self.idx[self.l:self.r]]  # (r, r-l)
            P_bar1 = self.U @ D_sqrt @ P_bar_star  # (p, r-l)
        else:
            P_bar1 = np.zeros((p, 0))
        
        # 计算 U_tilde (零特征值对应的特征向量)
        if p > self.r:
            U_tilde = self.U[:, self.r:] if self.r < p else np.zeros((p, 0))
        else:
            U_tilde = np.zeros((p, 0))
        
        self.P_bar = np.hstack([P_bar1, U_tilde])  # (p, p-l)
        
        # 计算 R_bar
        if self.r > self.l:
            R_bar1 = self.U @ D_sqrt_inv @ P_bar_star  # (p, r-l)
        else:
            R_bar1 = np.zeros((p, 0))
        
        self.R_bar = np.hstack([R_bar1, U_tilde])  # (p, p-l)
        
        # 计算协方差矩阵
        self.Sigma_eps = np.eye(self.l) - Sigma_hat_v  # (l, l)
        self.Sigma_eps_bar = np.block([
            [np.eye(max(self.r - self.l, 0)), np.zeros((max(self.r - self.l, 0), max(p - self.r, 0)))],
            [np.zeros((max(p - self.r, 0), max(self.r - self.l, 0))), np.zeros((max(p - self.r, 0), max(p - self.r, 0)))]
        ])  # (p-l, p-l)
        
        # 计算 Sigma_e
        # 计算协方差矩阵
        Y_s = Y_list[self.s]
        # 计算 V_list
        V_list = []
        for i in range(self.s + 1):
            V_i = Y_list[i] @ P_star  # (N, l)
            V_list.append(V_i)
        
        # Flatten self.B to get B_flat
        B_flat = self.B.reshape(-1, self.l)
        
        V_hat_s = np.hstack(V_list[:self.s]) @ B_flat
        y_hat = V_hat_s @ P_star.T @ D_sqrt @ self.U.T  # (N, p)
        e = y_centered[self.s:self.s+N, :] - y_hat  # (N, p)
        self.Sigma_e = np.cov(e.T)  # (p, p)
        
        return self
    
    def predict(self, y_test):
        """
        模型预测
        
        参数:
        - y_test: 测试数据 (s + N_test, p)
        
        返回:
        - y_pred: 预测值 (N_test, p)
        - v_pred: 潜变量预测值 (N_test, l)
        """
        if self.P is None or self.B is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 数据预处理
        y_test_centered = self.scaler.transform(y_test)
        N_test_total, p = y_test_centered.shape
        N_test = N_test_total - self.s
        
        if N_test <= 0:
            raise ValueError(f"测试数据长度必须大于VAR阶数s。当前: {N_test_total}, 需要至少: {self.s + 1}")
        
        # 计算标准化测试数据
        D_sqrt_inv = inv(np.sqrt(self.D))
        y_test_star = y_test_centered @ self.U @ D_sqrt_inv  # (N_test_total, r)
        
        # 初始化预测结果
        v_pred = np.zeros((N_test, self.l))
        y_pred = np.zeros((N_test, p))
        
        # 逐步预测
        for t in range(self.s, self.s + N_test):
            # 获取过去s个时间步的潜变量
            v_past = []
            for j in range(1, self.s + 1):
                # v_{t-j} = R^T y_{t-j}
                v_tj = self.R.T @ y_test_centered[t - j, :]  # (l,)
                v_past.append(v_tj)
            
            # 预测当前潜变量: v_hat_t = sum(B_j v_{t-j})
            v_hat_t = np.zeros(self.l)
            for j in range(self.s):
                v_hat_t += self.B[j] @ v_past[j]  # (l,)
            
            # 预测当前测量值: y_hat_t = P v_hat_t
            y_hat_t = self.P @ v_hat_t  # (p,)
            
            # 保存结果
            v_pred[t - self.s, :] = v_hat_t
            y_pred[t - self.s, :] = y_hat_t + self.scaler.mean_  # 反中心化
        
        return y_pred, v_pred
    
    def score(self, y_test):
        """
        计算模型性能指标
        
        参数:
        - y_test: 测试数据 (s + N_test, p)
        
        返回:
        - metrics: 性能指标字典
        """
        N_test_total, p = y_test.shape
        N_test = N_test_total - self.s
        
        # 获取预测值
        y_pred, v_pred = self.predict(y_test)
        y_true = y_test[self.s:self.s + N_test, :]
        
        # 计算均方根误差 (RMSE)
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        # 计算决定系数 (R²)
        ss_total = np.sum((y_true - np.mean(y_true, axis=0)) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        r2 = 1 - (ss_residual / ss_total)
        
        # 计算相似度指标 S (如论文中定义)
        def calculate_similarity(x, x_hat):
            numerator = np.sum(np.linalg.norm(x - x_hat, axis=1))
            denominator = np.sum(np.linalg.norm(x, axis=1) + np.linalg.norm(x_hat, axis=1))
            return 1 - (numerator / denominator)
        
        # 计算潜变量空间相似度
        v_true = self.R.T @ self.scaler.transform(y_test[self.s:self.s + N_test, :]).T  # (l, N_test)
        v_true = v_true.T  # (N_test, l)
        similarity_v = calculate_similarity(v_true, v_pred)
        
        # 计算测量空间相似度
        similarity_y = calculate_similarity(y_true, y_pred)
        
        metrics = {
            'rmse': rmse,
            'r2': r2,
            'similarity_v': similarity_v,
            'similarity_y': similarity_y,
            'prediction_error': ss_residual / N_test
        }
        
        return metrics
    
    def plot_dlv(self, y, title="Dynamic Latent Variables (DLVs)"):
        """
        绘制动态潜变量时间序列
        
        参数:
        - y: 输入数据 (N+s, p)
        - title: 图表标题
        """
        if self.R is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 计算潜变量
        y_centered = self.scaler.transform(y)
        v = self.R.T @ y_centered.T  # (l, N+s)
        v = v.T  # (N+s, l)
        
        # 绘制
        plt.figure(figsize=(12, 8))
        for i in range(self.l):
            plt.subplot(self.l, 1, i + 1)
            plt.plot(v[:, i])
            plt.title(f'{title} - DLV {i+1}')
            plt.xlabel('Time Step')
            plt.ylabel('Value')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_loadings(self, feature_names=None, title="DLV Loading Coefficients"):
        """
        绘制潜变量加载系数
        
        参数:
        - feature_names: 特征名称列表
        - title: 图表标题
        """
        if self.P is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        p, l = self.P.shape
        
        # 设置特征名称
        if feature_names is None:
            feature_names = [f'Feature {i+1}' for i in range(p)]
        
        # 绘制
        plt.figure(figsize=(12, 3 * l))
        for i in range(l):
            plt.subplot(l, 1, i + 1)
            # 区分正负值
            positive_idx = self.P[:, i] >= 0
            negative_idx = self.P[:, i] < 0
            
            plt.bar(np.arange(p)[positive_idx], self.P[positive_idx, i], 
                    color='darkblue', label='Positive')
            plt.bar(np.arange(p)[negative_idx], self.P[negative_idx, i], 
                    color='darkred', label='Negative')
            
            plt.title(f'{title} - DLV {i+1}')
            plt.xlabel('Features')
            plt.ylabel('Loading Coefficient')
            plt.xticks(np.arange(p), feature_names, rotation=45, ha='right')
            plt.legend()
            plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
    
    def plot_autocorrelation(self, y, max_lag=20, title="Autocorrelation of DLVs and Residuals"):
        """
        绘制潜变量和残差的自相关函数
        
        参数:
        - y: 输入数据 (N+s, p)
        - max_lag: 最大滞后阶数
        - title: 图表标题
        """
        if self.P is None or self.B is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 计算潜变量和残差
        y_centered = self.scaler.transform(y)
        N_total, p = y_centered.shape
        N = N_total - self.s
        
        # 计算潜变量
        v = self.R.T @ y_centered.T  # (l, N_total)
        v = v.T  # (N_total, l)
        
        # 计算残差
        y_pred, _ = self.predict(y)
        residuals = y_centered[self.s:self.s+N, :] - (y_pred - self.scaler.mean_)  # 反中心化后再中心化
        
        # 计算自相关函数
        def autocorrelation(x, max_lag):
            """计算自相关函数"""
            n, dim = x.shape
            acf = np.zeros((max_lag + 1, dim))
            
            for d in range(dim):
                for lag in range(max_lag + 1):
                    if lag == 0:
                        acf[lag, d] = 1.0
                    else:
                        acf[lag, d] = np.corrcoef(x[:-lag, d], x[lag:, d])[0, 1]
            
            return acf
        
        # 计算潜变量自相关
        v_acf = autocorrelation(v, max_lag)
        
        # 计算残差自相关（取前l个残差维度）
        res_acf = autocorrelation(residuals[:, :self.l], max_lag)
        
        # 绘制
        plt.figure(figsize=(12, 8))
        
        # 潜变量自相关
        plt.subplot(2, 1, 1)
        for i in range(self.l):
            plt.plot(range(max_lag + 1), v_acf[:, i], marker='o', label=f'DLV {i+1}')
        
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.axhline(y=1.96/np.sqrt(N_total), color='red', linestyle='--', alpha=0.5, label='95% CI')
        plt.axhline(y=-1.96/np.sqrt(N_total), color='red', linestyle='--', alpha=0.5)
        plt.title(f'{title} - DLVs')
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 残差自相关
        plt.subplot(2, 1, 2)
        for i in range(self.l):
            plt.plot(range(max_lag + 1), res_acf[:, i], marker='s', label=f'Residual {i+1}')
        
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.axhline(y=1.96/np.sqrt(N), color='red', linestyle='--', alpha=0.5, label='95% CI')
        plt.axhline(y=-1.96/np.sqrt(N), color='red', linestyle='--', alpha=0.5)
        plt.title(f'{title} - Residuals')
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def generate_lorenz_data(n_samples=10000, p=15, l=3, sigma=0.1):
    """
    生成基于Lorenz吸引子的仿真数据
    
    参数:
    - n_samples: 样本数
    - p: 测量维度
    - l: 潜变量维度
    - sigma: 噪声标准差
    
    返回:
    - y: 仿真数据 (n_samples, p)
    - v: 真实潜变量 (n_samples, l)
    - P: 真实加载矩阵 (p, l)
    - P_bar: 真实噪声加载矩阵 (p, p-l)
    """
    # Lorenz吸引子参数
    sigma_lorenz = 10.0
    beta = 8.0 / 3.0
    rho = 28.0
    
    # 生成Lorenz潜变量
    v = np.zeros((n_samples, l))
    v[0] = np.random.rand(l)  # 初始值
    
    for t in range(1, n_samples):
        dx = sigma_lorenz * (v[t-1, 1] - v[t-1, 0])
        dy = v[t-1, 0] * (rho - v[t-1, 2]) - v[t-1, 1]
        dz = v[t-1, 0] * v[t-1, 1] - beta * v[t-1, 2]
        
        v[t] = v[t-1] + 0.01 * np.array([dx, dy, dz])  # 时间步长0.01
    
    # 生成随机加载矩阵（正交）
    np.random.seed(42)
    P = np.random.randn(p, l)
    # 正交化
    P, _ = np.linalg.qr(P)
    
    # 生成噪声加载矩阵（与P正交）
    P_bar = np.random.randn(p, p - l)
    # 正交化并确保与P正交
    P_bar = P_bar - P @ P.T @ P_bar
    P_bar, _ = np.linalg.qr(P_bar)
    
    # 生成噪声
    eps_bar = sigma * np.random.randn(n_samples, p - l)
    
    # 生成测量数据 y = P v + P_bar eps_bar
    y = v @ P.T + eps_bar @ P_bar.T
    
    return y, v, P, P_bar

def main():
    """
    示例：使用Lorenz仿真数据训练和评估PredVAR模型
    """
    # 1. 生成仿真数据
    print("生成Lorenz仿真数据...")
    n_samples_total = 10000  # 总样本数
    s = 12  # VAR阶数
    l = 3  # 真实潜变量维度
    p = 15  # 测量维度
    
    y, v_true, P_true, P_bar_true = generate_lorenz_data(
        n_samples=n_samples_total,
        p=p,
        l=l,
        sigma=0.1
    )
    
    # 划分训练集和测试集
    train_ratio = 0.7
    n_train_total = int(train_ratio * n_samples_total)
    n_test_total = n_samples_total - n_train_total
    
    y_train = y[:n_train_total, :]
    y_test = y[n_train_total - s:, :]  # 测试集需要包含s个历史样本
    
    print(f"训练集大小: {y_train.shape}")
    print(f"测试集大小: {y_test.shape}")
    
    # 2. 初始化并训练模型
    print("\n训练PredVAR模型...")
    model = PredVAR(l=l, s=s, max_iter=50, tol=1e-8, verbose=True)
    model.fit(y_train)
    
    # 3. 模型评估
    print("\n模型评估...")
    metrics = model.score(y_test)
    print("测试集性能指标:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")
    
    # 4. 可视化结果
    print("\n生成可视化结果...")
    
    # 绘制潜变量
    model.plot_dlv(y_train, title="DLVs from Lorenz Data (Training Set)")
    
    # 绘制加载系数
    feature_names = [f'Measurement {i+1}' for i in range(p)]
    model.plot_loadings(feature_names=feature_names, title="DLV Loading Coefficients (Lorenz Data)")
    
    # 绘制自相关函数
    model.plot_autocorrelation(y_train, max_lag=20, title="Autocorrelation (Lorenz Data)")
    
    # 5. 预测示例
    print("\n预测示例...")
    y_pred, v_pred = model.predict(y_test)
    
    # 绘制前100个预测值与真实值的对比
    plt.figure(figsize=(12, 6))
    plt.plot(y_test[s:s+100, 0], label='True Value (Measurement 1)', linewidth=2)
    plt.plot(y_pred[:100, 0], label='Predicted Value (Measurement 1)', linestyle='--', linewidth=2)
    plt.title('Prediction vs True Value (First 100 Samples)')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # 计算真实潜变量与估计潜变量的相似度
    v_estimated_train = model.R.T @ model.scaler.transform(y_train).T
    v_estimated_train = v_estimated_train.T
    similarity_v_true = 1 - np.sum(np.linalg.norm(v_true[:n_train_total] - v_estimated_train, axis=1)) / \
                       np.sum(np.linalg.norm(v_true[:n_train_total], axis=1) + np.linalg.norm(v_estimated_train, axis=1))
    print(f"真实潜变量与估计潜变量的相似度: {similarity_v_true:.6f}")

if __name__ == "__main__":
    main()