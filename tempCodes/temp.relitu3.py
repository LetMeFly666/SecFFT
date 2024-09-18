'''
Author: LetMeFly
Date: 2024-09-13 23:38:43
LastEditors: LetMeFly
LastEditTime: 2024-09-13 23:39:17
'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 生成模拟数据
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

# 创建图像和子图
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))

# 标题和标签
attacks = ['Attack A', 'Attack B']
methods = ['Cosine Similarity Detection', 'FLTrust', 'FLAME', 'SecFFT']

# 生成热力图数据并绘制
for i in range(2):  # 两行，分别为Attack A和Attack B
    for j in range(4):  # 四列，分别为四种方法
        # 生成模拟数据
        scores = generate_score_data()
        ax = axes[i, j]
        
        # 生成一维热力图数据
        heatmap_data = scores.reshape(1, -1)
        
        # 绘制热力图
        sns.heatmap(heatmap_data, ax=ax, cmap='coolwarm', cbar=False, annot=True)
        
        # 设置标题和轴标签
        ax.set_title(f'{attacks[i]} - {methods[j]}', fontsize=12)
        ax.set_xlabel('Client Index', fontsize=10)
        ax.set_yticks([])  # 隐藏y轴刻度

# 调整子图布局
plt.tight_layout()

# 保存为jpg文件
plt.savefig('detection_results.jpg', format='jpg')
