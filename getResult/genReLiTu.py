'''
Author: LetMeFly
Date: 2024-09-13 23:43:25
LastEditors: LetMeFly
LastEditTime: 2024-09-14 01:52:38
'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# TODO: 使用真实数据
# 生成模拟数据的评分
def generate_score_data(num_clients=20, num_malicious=4):
    scores = np.random.rand(num_clients)  # 随机生成评分
    
    # 模拟恶意客户端和良性客户端的评分情况
    # 恶意客户端评分较高，良性客户端评分较低
    for i in range(num_clients):
        if i < num_malicious:  # 恶意客户端
            scores[i] = np.random.uniform(0.7, 1.0)
        else:  # 良性客户端
            scores[i] = np.random.uniform(0.0, 0.3)
    
    return scores

# 生成两两客户端对比矩阵 (这里使用绝对差值作为示例)
def generate_comparison_matrix(scores):
    num_clients = len(scores)
    comparison_matrix = np.zeros((num_clients, num_clients))
    
    # 计算客户端对之间的评分差异
    for i in range(num_clients):
        for j in range(num_clients):
            comparison_matrix[i, j] = abs(scores[i] - scores[j])  # 使用绝对差值
            
    return comparison_matrix

# 创建图像和子图
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))

# 标题和标签
attacks = ['NEUROTOXIN', 'MR']
methods = ['Cosine Similarity Detection', 'FLTrust', 'FLAME', 'SecFFT']

# 生成热力图数据并绘制
for i in range(2):  # 两行，分别为Attack A和Attack B
    for j in range(4):  # 四列，分别为四种方法
        # 生成模拟评分数据
        scores = generate_score_data()
        
        # 生成两两对比矩阵
        comparison_matrix = generate_comparison_matrix(scores)
        ax = axes[i, j]
        
        # 绘制热力图
        sns.heatmap(comparison_matrix, ax=ax, cmap='coolwarm', cbar=True, annot=False)
        
        # 设置标题和轴标签
        ax.set_title(f'{attacks[i]} - {methods[j]}', fontsize=12)
        ax.set_xlabel('Client Index', fontsize=10)
        ax.set_ylabel('Client Index', fontsize=10)

# 调整子图布局
plt.tight_layout()

# 保存为jpg文件
plt.savefig('detection_comparison_results.jpg', format='jpg')
