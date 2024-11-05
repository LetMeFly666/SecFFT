import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np

def plot_detection_heatmaps_3x4(*heatmap_params):
    """
    绘制三种攻击（原始 NEUROTOXIN、限制大小的攻击、限制角度和大小的攻击）下四种防御方法（Foolsgold, FLTrust, FLAME, SecFFT）的检测结果热力图。

    参数:
    - heatmap_params: 12 个参数，分别代表不同攻击下不同防御方法的评分或相似度矩阵。
                      每个参数可以是二维 torch.Tensor（torch.Size([50, 50])），表示已经计算好的相似度矩阵，
                      也可以是一维 torch.Tensor（torch.Size([50])），表示单个评分数组。
                      如果是一维数组，则函数会先计算出两两之间的相似度矩阵。
    """
    
    assert len(heatmap_params) == 12, "必须提供 12 个参数，每个参数表示一个子图的数据。"

    # 计算两两相似度矩阵的函数
    def compute_similarity_matrix(scores: torch.Tensor) -> np.ndarray:
        """
        接受一个一维的 torch.Tensor，并计算两两之间的相似度矩阵。

        参数:
        - scores: torch.Tensor, 一维张量，表示单个评分数组。

        返回:
        - similarity_matrix: np.ndarray, 二维数组，表示两两之间的相似度矩阵。
        """
        num_clients = len(scores)
        similarity_matrix = np.zeros((num_clients, num_clients))
        
        # 计算相似度矩阵
        for i in range(num_clients):
            for j in range(num_clients):
                if i == j:
                    similarity_matrix[i, j] = 1.0  # 自己与自己的相似度为1
                else:
                    similarity_matrix[i, j] = 1 - abs(scores[i].item() - scores[j].item())  # 使用简单的相似性计算
        
        return similarity_matrix

    # 计算每组热力图数据的全局最小值和最大值
    first_column_min_val, first_column_max_val = float('inf'), float('-inf')
    other_columns_min_val, other_columns_max_val = float('inf'), float('-inf')

    heatmap_arrays = []

    for idx, heatmap_data in enumerate(heatmap_params):
        # 如果是二维 tensor，直接转换为 numpy
        if len(heatmap_data.shape) == 2:
            if isinstance(heatmap_data, torch.Tensor):
                heatmap_array = heatmap_data.cpu().numpy()
            else:
                heatmap_array = heatmap_data
        # 如果是一维 tensor，计算相似度矩阵
        elif len(heatmap_data.shape) == 1:
            heatmap_array = compute_similarity_matrix(heatmap_data)
        else:
            raise ValueError("参数必须是 1D 或 2D 的 torch.Tensor。")
        
        # 根据图的位置更新不同的全局最小值和最大值
        if idx % 4 == 0:  # 第一列的图
            first_column_min_val = min(first_column_min_val, heatmap_array.min())
            first_column_max_val = max(first_column_max_val, heatmap_array.max())
        else:  # 其他列的图
            other_columns_min_val = min(other_columns_min_val, heatmap_array.min())
            other_columns_max_val = max(other_columns_max_val, heatmap_array.max())
        
        heatmap_arrays.append(heatmap_array)

    # 创建图像和子图
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20, 15))

    # 标题和标签
    attacks = ['Original NEUR', 'Size constrained attack', 'Angle and size constrained attack']
    methods = ['Foolsgold', 'FLTrust', 'Flame', 'SecFFT']

    # 生成热力图数据并绘制
    for i in range(3):  # 三行，分别为不同的攻击
        for j in range(4):  # 四列，分别为四种防御方法
            ax = axes[i, j]
            heatmap_array = heatmap_arrays[i * 4 + j]  # 获取每个参数对应的数据
            
            # 确定 vmin 和 vmax
            if j == 0:  # 第一列的图
                vmin = first_column_min_val
                vmax = first_column_max_val
            else:  # 其他列的图
                vmin = other_columns_min_val
                vmax = other_columns_max_val
            
            # 绘制热力图，使用统一的颜色条范围
            sns.heatmap(heatmap_array, ax=ax, cmap='coolwarm', cbar=True, annot=False, vmin=vmin, vmax=vmax)
            
            # 设置标题和轴标签
            ax.set_title(f'{attacks[i]} - {methods[j]}', fontsize=12)
            ax.set_xlabel('Client Index', fontsize=10)
            if j == 0:
                ax.set_ylabel('Client Index', fontsize=10)
            else:
                ax.set_yticks([])  # 隐藏y轴刻度

    # 调整子图布局
    plt.tight_layout()

    # 保存为pdf文件
    plt.savefig('detection_comparison_results_3x4.pdf', format='pdf')
    # plt.savefig('detection_comparison_results_3x4.png', format='png')  # 再保存一份为png

def generate_data_1dimension(num_clients, num_malicious):
    # 生成一维评分数组
    scores = torch.zeros(num_clients)
    for i in range(num_clients):
        if i < num_malicious:  # 恶意客户端
            scores[i] = torch.rand(1).item() * 0.3 + 0.7  # 评分在0.7到1之间
        else:  # 良性客户端
            scores[i] = torch.rand(1).item() * 0.3  # 评分在0到0.3之间
    return scores


if __name__ == '__main__':
    # 生成模拟数据
    num_clients = 50
    num_malicious = 20  # 前20个是恶意客户端

    # 生成12个模拟数据 (可以是1D的评分数组，也可以是计算好的2D相似度矩阵)
    data_params = []

    # 使用两种数据：随机生成的50x50相似度矩阵和一维评分数组
    for _ in range(12):
        if np.random.rand() > 0.5:
            # 生成二维相似度矩阵
            data_params.append(torch.rand(50, 50))
        else:
            # 生成一维评分数组
            data_params.append(generate_data_1dimension(num_clients, num_malicious))

    # 调用绘制函数
    plot_detection_heatmaps_3x4(*data_params)
