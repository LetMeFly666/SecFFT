修改这个函数
```
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

    # 创建图像和子图
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20, 15))

    # 标题和标签
    attacks = ['Original NEUR', 'Size constrained attack', 'Angle and size constrained attack']
    methods = ['Foolsgold', 'FLTrust', 'Flame', 'SecFFT']

    # 生成热力图数据并绘制
    for i in range(3):  # 三行，分别为不同的攻击
        for j in range(4):  # 四列，分别为四种防御方法
            ax = axes[i, j]
            heatmap_data = heatmap_params[i * 4 + j]  # 获取每个参数对应的数据
            
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
            
            # 绘制热力图
            sns.heatmap(heatmap_array, ax=ax, cmap='coolwarm', cbar=True, annot=False)
            
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
    plt.savefig('detection_comparison_results_3x4.pdf', format='pdf')


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
```
将所有子图度量衡范围统一






第一列三张图度量衡范围保持统一，后三列九张图保持统一





我这个表太宽了，超出页面范围了，应该怎么调
```
\begin{table*}[h!]
\small
\centering
\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|c|c|c|}
\hline
Attack strategy & Defense & TP & FP & TN & FN & Accuracy & Precision & Recall & Specificity & FPR & FNR & AUC & MCC \\ \hline

\multirow{4}{*}{No strategy} & Foolsgold & 10 & 0 & 35 & 5 & 0.90 & 1.00 & 0.67 & 1.00 & 0.00 & 0.33 & 0.83 & 0.76 \\ \cline{2-14}
                      & FLTrust   & 6  & 0 & 35 & 9 & 0.82 & 1.00 & 0.40 & 1.00 & 0.00 & 0.60 & 0.70 & 0.56 \\ \cline{2-14}
                      & Flame     & 10 & 0 & 35 & 5  & 0.90 & 1.00 & 0.67 & 1.00 & 0.00 & 0.33 & 0.83 & 0.76 \\ \cline{2-14}
                      & SecFFT    & \textbf{15} & 0 & 35 & 0  & \textbf{1.00} & 1.00 & \textbf{1.00} & 1.00 & 0.00 & 0.00 & \textbf{1.00} & \textbf{1.00} \\ \hline
\multirow{4}{*}{Size-limited strategy} & Foolsgold & 12 & 1 & 34 & 3 & 0.92 & 0.92 & 0.80 & 0.97 & 0.03 & 0.20 & 0.88 & 0.72 \\ \cline{2-14}
                      & FLTrust   & 8  & 2 & 33 & 7 & 0.82 & 0.80 & 0.53 & 0.94 & 0.06 & 0.47 & 0.74 & 0.61 \\ \cline{2-14}
                      & Flame     & 11 & 0 & 35 & 4  & 0.92 &1.00 & 0.73 & 1.00 & 0.00 & 0.27 & 0.87 & 0.78 \\ \cline{2-14}
                      & SecFFT    & \textbf{14} & 0 & 35 & \textbf{1}  & \textbf{0.98} & 1.00 & \textbf{0.93} & 1.00 & 0.00 & 0.07 & \textbf{0.96} & \textbf{0.91} \\ \hline

\multirow{4}{*}{Angle-limited strategy} & Foolsgold & 11 & 2 & 33 & 4 & 0.88 & 0.85 & 0.73 & 0.94 & 0.06 & 0.27 & 0.84 & 0.68 \\ \cline{2-14}
                       & FLTrust   & 9  & 1 & 34 & 6 & 0.86 & 0.90 & 0.60 & 0.97 & 0.03 & 0.40 & 0.80 & 0.65 \\ \cline{2-14}
                       & Flame     & 10 & 0 & 35 & 5  & 0.90 & 1.00 & 0.67 & 1.00 & 0.00 & 0.33 & 0.83 & 0.76 \\ \cline{2-14}
                       & SecFFT    & \textbf{13} & 0 & 35 & \textbf{2}  & \textbf{0.96} & 1.00 & \textbf{0.87} & 1.00 & 0.00 & 0.13 & \textbf{0.94} & \textbf{0.88} \\ \hline
\end{tabular}
\caption{Classification metrics results under different attack and defense methods}
\end{table*}
```




Attack strategy、No strategy等第一列这些字可不可以换行