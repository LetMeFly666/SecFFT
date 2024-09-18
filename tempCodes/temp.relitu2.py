'''
Author: LetMeFly
Date: 2024-09-13 22:56:00
LastEditors: LetMeFly
LastEditTime: 2024-09-13 23:07:43
'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 生成模拟数据
def generate_heatmap_data(num_clients=20, num_malicious=4):
    data = np.random.rand(num_clients, num_clients)  # 随机生成数据
    
    # 模拟恶意客户端和良性客户端的识别情况
    # 恶意客户端的相似度较高，良性客户端的相似度较低
    for i in range(num_clients):
        for j in range(num_clients):
            if i < num_malicious and j < num_malicious:  # 恶意客户端之间的高相似度
                data[i, j] = np.random.uniform(0.7, 1.0)
            elif i >= num_malicious and j >= num_malicious:  # 良性客户端之间的低相似度
                data[i, j] = np.random.uniform(0.0, 0.3)
            else:  # 恶意和良性之间的相似度中等
                data[i, j] = np.random.uniform(0.3, 0.7)
    # print(data)
    # exit(0)
    return data

# 创建图像和子图
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))

# 标题和标签
attacks = ['Attack A', 'Attack B']
methods = ['Cosine Similarity Detection', 'FLTrust', 'FLAME', 'SecFFT']

# 生成热力图数据并绘制
for i in range(2):  # 两行，分别为Attack A和Attack B
    for j in range(4):  # 四列，分别为四种方法
        # 生成模拟数据
        heatmap_data = generate_heatmap_data()
        ax = axes[i, j]
        
        # 绘制热力图
        sns.heatmap(heatmap_data, ax=ax, cmap='coolwarm', cbar=False)
        
        # 设置标题
        ax.set_title(f'{attacks[i]} - {methods[j]}', fontsize=12)
        ax.set_xlabel('Client Index', fontsize=10)
        ax.set_ylabel('Client Index', fontsize=10)

# 调整子图布局
plt.tight_layout()

# 保存为jpg文件
plt.savefig('temp.relitu.2x4.jpg', format='jpg')
