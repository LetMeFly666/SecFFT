'''
Author: LetMeFly
Date: 2024-09-13 23:43:25
LastEditors: LetMeFly
LastEditTime: 2024-09-14 01:52:38
'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 生成模拟数据的评分
def generate_score_data(num_clients=50, num_malicious=10):
    scores = np.random.rand(num_clients)  # 随机生成评分
    
    # 模拟恶意客户端和良性客户端的评分情况
    # 恶意客户端评分较高，良性客户端评分较低
    for i in range(num_clients):
        if i < num_malicious:  # 恶意客户端
            scores[i] = np.random.uniform(0.7, 1.0)
        else:  # 良性客户端
            scores[i] = np.random.uniform(0.0, 0.3)
    
    return scores

# 生成两两客户端对比矩阵 (这里使用余弦相似度作为示例)
def generate_cosine_similarity_matrix(scores):
    num_clients = len(scores)
    similarity_matrix = np.zeros((num_clients, num_clients))
    
    # 计算客户端对之间的余弦相似度
    for i in range(num_clients):
        for j in range(num_clients):
            if i == j:
                similarity_matrix[i, j] = 1.0  # 自己与自己的相似度为1
            else:
                similarity_matrix[i, j] = 1 - abs(scores[i] - scores[j])  # 使用简单的相似性计算
                
    return similarity_matrix

# 生成其他检测方法的评分数组 (FLTrust, FLAME, SecFFT) 并计算相似度矩阵
def generate_defense_scores_similarity_matrix(num_clients=50, num_malicious=10):
    scores = np.random.rand(num_clients)  # 随机生成评分
    
    # 模拟恶意客户端和良性客户端的评分情况
    for i in range(num_clients):
        if i < num_malicious:  # 恶意客户端
            scores[i] = np.random.uniform(0.7, 1.0)
        else:  # 良性客户端
            scores[i] = np.random.uniform(0.0, 0.3)
    
    # 计算两两客户端之间的相似度矩阵（这里也用相似度差异作为计算示例）
    similarity_matrix = np.zeros((num_clients, num_clients))
    for i in range(num_clients):
        for j in range(num_clients):
            if i == j:
                similarity_matrix[i, j] = 1.0  # 自己与自己的相似度为1
            else:
                similarity_matrix[i, j] = 1 - abs(scores[i] - scores[j])  # 使用简单的相似性计算
    
    return similarity_matrix

# 创建图像和子图
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))

# 标题和标签
attacks = ['Attack A', 'Attack B']
methods = ['Cosine Similarity Detection', 'FLTrust', 'FLAME', 'SecFFT']

# 生成热力图数据并绘制
for i in range(2):  # 两行，分别为Attack A和Attack B
    for j in range(4):  # 四列，分别为四种方法
        ax = axes[i, j]
        
        if j == 0:  # 第一列为余弦相似度 (50x50矩阵)
            scores = generate_score_data(num_clients=50, num_malicious=10)
            comparison_matrix = generate_cosine_similarity_matrix(scores)
            sns.heatmap(comparison_matrix, ax=ax, cmap='coolwarm', cbar=True, annot=False)
        else:  # 其余为根据评分计算的两两相似度矩阵
            defense_scores_similarity_matrix = generate_defense_scores_similarity_matrix(num_clients=50, num_malicious=10)
            sns.heatmap(defense_scores_similarity_matrix, ax=ax, cmap='coolwarm', cbar=True, annot=False)
        
        # 设置标题和轴标签
        ax.set_title(f'{attacks[i]} - {methods[j]}', fontsize=12)
        ax.set_xlabel('Client Index', fontsize=10)
        if j == 0:
            ax.set_ylabel('Client Index', fontsize=10)
        else:
            ax.set_yticks([])  # 隐藏y轴刻度

# 调整子图布局
plt.tight_layout()

# 保存为jpg文件
plt.savefig('detection_comparison_results.jpg', format='jpg')
