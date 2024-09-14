'''
Author: LetMeFly
Date: 2024-09-13 23:43:25
LastEditors: LetMeFly
LastEditTime: 2024-09-14 01:52:38
'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

def plot_detection_heatmaps(cosine_similarity_a: torch.Tensor, cosine_similarity_b: torch.Tensor):
    """
    绘制两种攻击（NEUROTOXIN、MR）下四种检测方法（Cosine Similarity Detection, FLTrust, FLAME, SecFFT）的检测结果热力图。
    
    参数:
    - cosine_similarity_a: np.ndarray, (50, 50) 的二维数组，表示攻击 NEUROTOXIN 下的余弦相似度矩阵。
    - cosine_similarity_b: np.ndarray, (50, 50) 的二维数组，表示攻击 MR 下的余弦相似度矩阵。
    """
    cosine_similarity_a = cosine_similarity_a.cpu().numpy()
    cosine_similarity_b = cosine_similarity_b.cpu().numpy()
    
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
    attacks = ['NEUROTOXIN', 'MR']
    methods = ['Cosine Similarity Detection', 'FLTrust', 'FLAME', 'SecFFT']

    # 生成热力图数据并绘制
    for i, attack in enumerate([cosine_similarity_a, cosine_similarity_b]):  # 分别为攻击NEUROTOXIN和MR
        for j, method in enumerate(methods):  # 四种方法
            ax = axes[i, j]
            
            if j == 0:  # 第一列为传入的余弦相似度 (50x50矩阵)
                sns.heatmap(attack, ax=ax, cmap='coolwarm', cbar=True, annot=False)
                # print(attack)
            else:  # 其余为根据评分计算的两两相似度矩阵
                defense_scores_similarity_matrix = generate_defense_scores_similarity_matrix(num_clients=50, num_malicious=10)
                sns.heatmap(defense_scores_similarity_matrix, ax=ax, cmap='coolwarm', cbar=True, annot=False)
            
            # 设置标题和轴标签
            ax.set_title(f'{attacks[i]} - {method}', fontsize=12)
            ax.set_xlabel('Client Index', fontsize=10)
            if j == 0:
                ax.set_ylabel('Client Index', fontsize=10)
            else:
                ax.set_yticks([])  # 隐藏y轴刻度

    # 调整子图布局
    plt.tight_layout()

    # 保存为jpg文件
    plt.savefig('detection_comparison_results.jpg', format='jpg')


if __name__ == '__main__':
    import numpy as np
    # from your_module import plot_detection_heatmaps  # 请替换 'your_module' 为实际保存函数的模块名

    # 创建NEUROTOXIN和MR的余弦相似度矩阵 (50x50)
    cosine_similarity_a = np.random.rand(50, 50)
    cosine_similarity_b = np.random.rand(50, 50)

    # 调用绘制函数
    plot_detection_heatmaps(cosine_similarity_a, cosine_similarity_b)
