<!--
 * @Author: LetMeFly
 * @Date: 2024-08-18 10:06:39
 * @LastEditors: LetMeFly
 * @LastEditTime: 2024-09-08 03:04:35
-->




```
Line 7: Char 1: syntax error: non-declaration statement outside function body (solution.go)
Line 8: Char 1: syntax error: imports must appear before other declarations (solution.go)
```

```
package main;
import "unicode"

func clearDigits(s string) string {
    ansList := []rune{}
    cntDigit := 0
    for i := len(s) - 1; i >= 0; i-- {
        if unicode.IsDigit(rune(s[i])) {
            cntDigit++
        } else if cntDigit > 0 {
            cntDigit--
        } else {
            ansList = append(ansList, rune(s[i]))
        }
    }
    for i := 0; i < len(ansList) / 2; i++ {
        ansList[i], ansList[len(ansList) - i - 1] = ansList[len(ansList) - i - 1], ansList[i]
    }
    return string(ansList)
}
```





PPT重复上次操作快捷键




015.ChatLogWithGPT.审稿意见撰写（未删除）.cb86df59-e452-4fc1-a99c-dae4e07da0cd.json.审稿意见2.txt




英语动词变复数



reactionary为什么要变成reactionaries




在以下字符串中提取第二个时间，并转为时间戳`全局优惠生效中，2024-09-03 00:00:00 ~ 2024-09-06 00:00:00 期间，全站`





用python画这种梯度下降图的svg





```
\subsection{Defender's Goal}
模型性能，鲁棒性，有效性
\section{Method}
本节将仔细说明本文的主要思想以及详细的设计。此外，为了清晰且易于理解，表\ref{tabl:notations}中总结了本文中使用过的符号及相应的解释。

\begin{table*}[h]
\centering
\caption{Notations and Corresponding Definitions}
\label{tabl:notations}
\begin{tabular}{|c|l|}
\hline
\textbf{Notation} & \textbf{Definition} \\ \hline
% 联邦学习
$ n $ & The number of the nodes participating during each round. \\ \hline
% $\mathcal{M}$ & The number of the participants during each round. \\ \hline
$D_i$ & The private local dataset of node $C_i$. \\ \hline

% 单论识别
$G^t$ & The global model at round $t$. \\ \hline
$\theta_i^t$ & The local model of participant at round $t$. \\ \hline
$\triangledown \theta_i^t$ & The local update of node $R_i$ during the $t$-th round. \\ \hline
$G_i^t $ & The low-frequency components extracted from the update of node $R_i$ during the $t$-th round.. \\ \hline
$m$ & The length of the low-frequency components. \\ \hline
$H^t$ & The matrix formed by stacking $G_i^t$, with a shape of $m \times n$ during the $t$-th round. \\ \hline
$\tilde{G}_{0}^t $ & The clean ingredient extracted from the nodes' low-frequency components during the $t$-th round. \\ \hline
$\lambda_{max}^t$ & The largest singular value of matrix $H^t$ during the $t$-th round. \\ \hline
$\xi_{max}^t$ & The corresponding left singular vector of singular value $\lambda_{max}^t$ during the $t$-th round. \\ \hline
$P_i^t$ & The distribution corresponding to $G_i^t$ after applying the Softmax function during the $t$-th round.. \\ \hline
$Q^t$ &  The distribution corresponding to $\tilde{G}_{0}^t$ after applying the Softmax function during the $t$-th round.. \\ \hline

% 待总结

\end{tabular}
\end{table*}

\subsection{单轮攻击识别}
\textbf{Motivation}: 后门攻击
% 后门攻击的性质：后门攻击在于使得模型建立起触发器和标签之间的关联，而不是常规图像之间的关联。
% 为什么考虑转换到频率维度，现有的方案存在哪些缺点。

\textbf{Overview}: 从模型更新的频率入手，利用DCT进行转换，提取出干净的频率分布样本，最后利用KL散度（HDBSCAN聚类）进行恶意用户识别。

\begin{algorithm}
\caption{Malicious Node Detection via DCT}
\label{alg:malicious-node-detection}
\begin{algorithmic}[1]
\State \textbf{Input:} $d$, $n$, $(\triangledown \theta_0^t, \triangledown \theta_1^t, \ldots, \triangledown \theta_n^t) \in \mathbb{R}^{n \times d}$, $m$ \Comment{$d$ is the dimension of each node update; $n$ is the number of the nodes participating during each round; $(\triangledown \theta_0^t, \triangledown \theta_1^t, \ldots, \triangledown \theta_n^t) \in \mathbb{R}^{n \times d}$is the local updates from nodes during the $t$-th round; $m$ is the length of low-frequency components}
\State \textbf{Output:} $U_{nor}$, $U_{mal}$ \Comment{benign nodes, malicious nodes}

\State Step 1: Frequency Domain Transformation
\For {Node $R_i$}
    \State $\bar{\theta_i^t} \gets Flatten(\triangledown \theta_i^t)$
    \For{$k = 0$ to $d-1$}
        \State $G_i^t[k] \gets 0$
        \For{$j = 0$ to $d-1$}
            \State $G_i^t[k] \gets G_i^t[k] + \bar{\theta_i^t}[j] \cos \left( \frac{\pi}{d} \left(j + \frac{1}{2}\right) k \right)$
        \EndFor
    \EndFor
\EndFor

\State

\State Step 2: Clean Ingredient Extraction
\State $H^t \gets (G_0^t, G_1^t, \ldots, G_{\mathcal{M}}^t) \in \mathbb{R}^{d \times m}$ \Comment{Stacking to form Matrix $H^t$.}

\State $\hat{H}^t \gets (H^t)^{T} H^t$
\State $\lambda_{max}^t, \xi_{max}^t \gets \text{eig}(\hat{H}^t)$ \Comment{Calculating the maximum singular value and its corresponding eigenvector}
\State $\tilde{G}^t_0 \gets \frac{H^t \xi_{max}^t}{\sqrt{\lambda_{max}^t}}$ \Comment{The clean ingredient}

\State

\State Step 3: Kullback-Leibler Divergence Calculation
\State $P_i^t \gets Softmax(G_i^t), \, Q^t \gets Softmax(\tilde{G}^t_0)$
\For {Node $R_i$}
    \State $KL_i \gets \sum_{k=0}^{m-1} P_i^t[k] \log \left( \frac{P_i^t[k]}{Q^t[k]} \right)$
\EndFor
\State $S \gets \{KL_1, KL_2, \ldots, KL_n\}$ \Comment{The distribution differences calculated by KL divergence.}

\State

\State Step 4: Malicious Node Detection
\State $\{C_1, C_2, \ldots, C_K\} \gets \text{HDBSCAN}(S)$
\State $U_{nor} \gets C_{\text{max}}$, $U_{mal} \gets \{C_i \, | \, i \neq \text{max}\}$

\end{algorithmic}
\end{algorithm}
算法\ref{alg:malicious-node-detection}展示了整体的恶意用户识别过程，主要包含四个步骤：Frequency Domain Transformation, Clean Ingredient Extraction, Kullback-Leibler Divergence Calculation,Single-round Malicious Node Detection。
\begin{enumerate}
\item \textbf{Frequency Domain Transformation}

Frequency Domain Transformation的主要过程如算法\ref{alg:malicious-node-detection}的4-12行所示。假设每个本地模型的更新为 \(\triangledown \theta_i^t \in \mathbb{R}^d\)，其中 \(i\) 表示机器人节点，\(t\) 表示通信轮次，\(d\) 为本地模型上传更新的维度。对于每个更新 \(\triangledown \theta_i^t\)，我们通过一维的DCT-II将其转换为对应的频率分布，并提取对应的m个低频成分，得到低频向量\(G_i^t\)。对应的计算过程如下：
\begin{equation}
G_i^t = Trunc(DCT(Flatten(\triangledown \theta_i^t), m)
\end{equation}

\begin{equation}
Flatten: \bar{\theta_i^t} = Flatten(\triangledown \theta_i^t)
\end{equation}

\begin{equation}
DCT: G_i^t[k] = \sum_{j=0}^{d-1} \bar{\theta_i^t}[j] \cos \left( \frac{\pi}{d} \left(j + \frac{1}{2}\right) k \right)
\end{equation}

\item \textbf{Clean Ingredient Extraction}

Clean Ingredient Extraction的主要过程如算法\ref{alg:malicious-node-detection}的15-18行所示。将所有节点的低频向量堆叠为矩阵 \(H^t = ({G_0^t}, {G_1^t}, \ldots, {G_{\mathcal{M}}^t}) \in \mathbb{R}^{m \times n}\)，其中每一列代表一个节点的低频向量。通过计算矩阵的最大奇异值及其对应的左奇异向量来作为相应的干净成分。即计算出矩阵\(\hat{H}^t = (H^t)^{T} H^t\)最大特征值 \(\lambda_{\text{max}}\) 及其对应的特征向量 \(\xi_{\text{max}}\)，根据公式\ref{equation:singular}，得到干净向量 \(\tilde{G}_0^t\)：

\begin{equation}
\tilde{G}^t_0 = \frac{H^t \xi_{\text{max}}}{\sqrt{\lambda_{\text{max}}}}
\label{equation:singular}
\end{equation}

\item \textbf{Kullback-Leibler Divergence Calculation}

Kullback-Leibler Divergence Calculation的主要过程如算法\ref{alg:malicious-node-detection}的21-25行所示。对于每个节点的低频向量\(\mathbf{G}_i^t\)以及干净以及上一步求得的干净成分\(\tilde{G}^t_0\)，通过Softmax函数来将它们转换为相应的概率分布 \(P_i^t\) 和 \(Q^t\) 。然后利用KL散度即公式\ref{equation:KL}计算每个节点概率分布\((P_i^t\)和干净分量概率分布 \(Q^t\) 之间的距离，得到相应的分布差异集合\(S = {{KL}}_1, {{KL}}_2, \ldots, {{KL}}_n\)。

\begin{equation}
{{KL}}_i = D(P_i^t \| Q^t) = \sum_{k=0}^{m-1} P_i^t[k] \log \left( \frac{P_i^t[k]}{Q^t[k]} \right)
\label{equation:KL}
\end{equation}

其中
\[
P_i^t = Softmax(G_i^t)
\]

\[
Q^t = Softmax(\tilde{G_0^t})
\]

\item \textbf{Single-round Malicious Node Detection}

最后Single-round Malicious Node Detection的的主要过程如算法\ref{alg:malicious-node-detection}的28-29行所示。利用HDBSCAN对计算出来的KL散度集合$S$进行聚类，得到簇\(C_1, C_2, \ldots, C_K\)，从所有簇中选出最大的一簇（即包含节点数量最多的簇），记作 \(C_{max}\)。基于每轮参与训练的良性节点相较于恶意节点数量较多的假设，并且相较于恶意节点，所有良性节点与干净分量之间的KL散度具有更高的相似性，因此将 \(C_{max}\)中的所有节点标记为良性节点， 记作为\(U_{nor}\)。对于除\(C_{max}\)外的所有其他簇\(C_i\)（\(i \neq {max}\)），将其中所有节点标记为恶意节点\(U_{mal}\)。 

\end{enumerate}

\subsection{多轮次攻击识别}

\textbf{Motivation}: 后门攻击

\textbf{Overview}: 前面的攻击中已经可以有效地针对单轮次攻击进行识别，本部分主要结合客户端的历史记录对客户端进行有效的识别。本部分将保留历史$T$轮次的全局模型$G_t$，并将其Flatten后视为高维空间中的一个点；同时保留每个客户端的历史梯度变化，将其视为高维空间中的一个向量，将其延伸后视为一个射线。对于每个客户端，使用最小覆盖球算法计算得到一个最小的超球，覆盖掉$\varepsilon$比例的射线。最终球心的位置可以被视为对应客户端对全局模型的意图引导点，球心的半径可以视为对应客户端的置信度。

\begin{algorithm}
\caption{Malicious Node Detection via History}
\label{alg:malicious-node-detection-history}
\begin{algorithmic}[1]
\State \textbf{Input:} TODO:
\State Step 1: Keep T-rounds History
\For {Node $l_i$}
    \State $\bar{\theta_i^t} \gets Flatten(\triangledown \theta_{l_i}^t)$
    \State $DB.gradient\ \ +=\ \ \bar{\theta_i^t}$
    \State $DB.model\ \ +=\ \ G_{t-1}$
\EndFor
\end{algorithmic}
\end{algorithm}

算法\ref{alg:malicious-node-detection-history}展示了根据每个客户端历史记录进行识别的识别过程，主要包含TODO:个步骤：

\begin{enumerate}

\item \textbf{Keep T-rounds History}

“Keep T-rounds History”的主要算法过程如算法\ref{alg:malicious-node-detection-history}的TODO:行所示。像正常的联邦学习过程一样，中央服务器每次下发一个全局模型，每个客户端得到全局模型后使用本地数据进行训练，并将梯度变化上传到中央服务器中。我们将第$t$轮次的全局模型记为$G_t$，将客户端$l_i$第$t$轮次训练后的模型记为$\theta^t_{l_i}$。客户端$l_i$将训练后的模型$\theta^t_{l_i}$减去训练前中央服务器下发的全局模型$G_t$，就得到了$t$轮次的梯度变化$\triangledown \theta^t_{l_i}=\theta^t_{l_i}-G_{t-1}$。客户端将梯度变化$\triangledown \theta^t_{l_i}$上传到中央服务器，中央服务器在聚合的同时，记录下每个客户端当前轮次的梯度变化$\triangledown \theta^t_{l_i}$展平后的结果$\bar{\theta_i^t}$，同时存下中央服务器上次下发的全局模型$G_{t-1}$。中央服务器最多保留$T$轮次的模型和梯度历史记录。此部分可以参考算法\ref{alg:malicious-node-detection-history}的step1。

\item \textbf{Obtain the Purpose Intention}

“Obtain the Purpose Intention”的主要算法过程如算法\ref{alg:malicious-node-detection-history}的TODO:行所示。这个问题可以抽象为高维空间中的一些具有起点的射线。射线的起点代表$t-1$轮次的全局模型，射线的方向代表$t$轮次展平后的梯度变化。问题的优化目标是：找到一个最小的超球，包含至少$\varepsilon$比例的射线。

$TODO$

具体来说，根据

\item \textbf{Obtain the Purpose Intention}

\end{enumerate}
```

严格按照已有格式，扩写`$TODO$`这一部分的内容。

要求包含一步一步的具体描述及其公式。





先返回`构建射线模型`部分和`最小覆盖球的求解`部分的latex源码，这时还不需要写有关置信度计算的内容。

注意，要和之前的章节保持一致，不能突兀。




修改这一行，让横线覆盖整个参数
```
\item 设 \( G_{t-T}, G_{t-T+1}, \dots, G_{t-1} \) 是最近 \(T\) 轮次的全局模型，且这些全局模被“展平”到了高维空间中的一个点。类似地，设 \( \bar{\theta_{l_i}^{t-T}}, \bar{\theta_{l_i}^{t-T+1}}, \dots, \bar{\theta_{l_i}^{t-1}} \) 是最近 \(T\) 轮次客户端 \(l_i\) 上传的展平后的梯度变化向量。
```




找最小覆盖球的部分 写一个s.t.

只需要返回这个公式即可




返回这个公式的latex源码。其中s.t.要放到下一行




这个公式的s.t.还是在同一行显示了
```
\[
\min_{O, r} \, r \\
\text{s.t.} \quad \| O - (P_i + t_i \mathbf{d}_i) \|^2 \leq r^2, \, \forall i, \, t_i \geq 0
\]
```



写一个对T条射线按照距离起点距离排序的公式，并返回其latex





```
\item \textbf{Keep T-rounds History}

“Keep T-rounds History”的主要算法过程如算法\ref{alg:malicious-node-detection-history}的TODO:行所示。像正常的联邦学习过程一样，中央服务器每次下发一个全局模型，每个客户端得到全局模型后使用本地数据进行训练，并将梯度变化上传到中央服务器中。我们将第$t$轮次的全局模型记为$G_t$，将客户端$l_i$第$t$轮次训练后的模型记为$\theta^t_{l_i}$。客户端$l_i$将训练后的模型$\theta^t_{l_i}$减去训练前中央服务器下发的全局模型$G_t$，就得到了$t$轮次的梯度变化$\triangledown \theta^t_{l_i}=\theta^t_{l_i}-G_{t-1}$。客户端将梯度变化$\triangledown \theta^t_{l_i}$上传到中央服务器，中央服务器在聚合的同时，记录下每个客户端当前轮次的梯度变化$\triangledown \theta^t_{l_i}$展平后的结果$\bar{\theta_i^t}$，同时存下中央服务器上次下发的全局模型$G_{t-1}$。中央服务器最多保留$T$轮次的模型和梯度历史记录。此部分可以参考算法\ref{alg:malicious-node-detection-history}的step1。

\item \textbf{Obtain the Purpose Intention}

“Obtain the Purpose Intention”的主要算法过程如算法\ref{alg:malicious-node-detection-history}的TODO:行所示。这个问题可以抽象为高维空间中的一些具有起点的射线。射线的起点代表$t-1$轮次的全局模型，射线的方向代表$t$轮次展平后的梯度变化。问题的优化目标是：找到一个最小的超球，包含至少$\varepsilon$比例的射线。此部分又可以分为“Construct Ray Model”和“Minimum Enclosing Hypersphere Calculation”两部分。

“Construct Ray Model”的主要过程如下。对于每个客户端 \(l_i\)，中央服务器保留了最近 \(T\) 轮的全局模型历史以及每轮次的梯度更新历史。

\begin{itemize}
    \item 设 \( G_{t-T}, G_{t-T+1}, \dots, G_{t-1} \) 是最近 \(T\) 轮次的全局模型，且这些全局模被“展平”到了高维空间中的一个点。类似地，设 \( \overline{\theta_{l_i}^{t-T}}, \overline{\theta_{l_i}^{t-T+1}}, \dots, \overline{\theta_{l_i}^{t-1}} \) 是最近 \(T\) 轮次客户端 \(l_i\) 上传的展平后的梯度变化向量。
    \item 对于每一轮 \( t-j \) (\( j = 0, 1, \dots, T-1 \))，我们将每个射线的起点表示为展平后的全局模型 \( G_{t-j-1} \)，并将每个射线的方向向量表示为展平后的梯度变化 \( \overline{\theta}_{l_i}^{t-j} \)。因此，射线模型可以构建如下：

    \begin{equation}
    \overline{\theta_{l_i}^{t-j}} = Flatten(G_{t-j-1}),
    \end{equation}

    \begin{equation}
    \mathbf{d}_{l_i}^{t-j} = \bar{\theta}_{l_i}^{t-j},
    \end{equation}

    其中，\( \overline{\theta_{l_i}^{t-j}} \) 是射线的起点，\( \mathbf{d}_{l_i}^{t-j} \) 是射线的方向向量。
\end{itemize}

“Minimum Enclosing Hypersphere Calculation”的主要过程如下。基于构建的射线模型，使用最小覆盖球算法找到一个能够覆盖至少 \( \varepsilon \) 比例的射线的最小超球。

\begin{itemize}
    \item 首先，设定初始球心 \( O_0 \) 为所有射线起点的几何中心，计算公式如下：

    \begin{equation}
    O_0 = \frac{1}{T} \sum_{j=0}^{T-1} \overline{\theta_{l_i}^{t-T+1}}.
    \end{equation}

    \item 初始半径 \( r_0 \) 设定为从初始球心 \( O_0 \) 到所有射线起点的最大距离：

    \begin{equation}
    r_0 = \max_{1\leq T} \|O_0 - \overline{\theta_{l_i}^{t-T+1}}\|.
    \end{equation}

    \item 接下来，采用迭代方法来更新球心和半径：
    \begin{itemize}
        \item 对于每次迭代 \( k \)，计算当前球心 \( O_k \) 到所有射线的最近投影点 \( Q_{l_i}^{t-j} \)，按照距离当前球心的位置排序

        \begin{equation}
        \text{Sort}(\{ \| O - P_i \| : i = 1, 2, \ldots, T \})
        \end{equation}

        并计算这些投影点的最大距离 \( D_k \)：

        \begin{equation}
        D_k = \max_{1\leq j\leq \lceil n\times \varepsilon\rceil} \|O_k - Q_{l_i}^{t-j}\|.
        \end{equation}

        \begin{equation}
        O_{k+1} = O_k + \alpha \left(\frac{1}{\lceil n\times \varepsilon\rceil} \sum_{1\leq j\leq \lceil n\times \varepsilon\rceil} (Q_{l_i}^{t-j} - O_k)\right),
        \end{equation}

        \begin{equation}
        r_{k+1} = D_k,
        \end{equation}

        其中，\( \alpha \) 是学习率，通常取一个小的正数（如 0.1）。
    \end{itemize}
    \item 终止条件：当更新后的半径 \( r_{k+1} \) 与上一次的半径 \( r_k \) 相差小于一个预设的阈值 \( \epsilon \)（如 \( 10^{-6} \)），或者达到预设的最大迭代次数时，停止迭代。
\end{itemize}

这部分可以定义为一个有约束的最优化问题：

\[
\min_{O, r} \, r 
\]

\[
\text{s.t.} \quad \| O - (P_i + t_i \mathbf{d}_i) \|^2 \leq r^2, \, \forall i, \, t_i \geq 0
\]

\item \textbf{Obtain the Purpose Intention}

\end{enumerate}
```

这是我找超球这部分的代码，其中优化问题没有体现$\varepsilon$，请修改优化问题的公式描述`s.t.`这一部分。





能不能写成`_{1\leq j\leq \lceil n\times \varepsilon\rceil}`的形式









接下来写异常检测部分。

我的置信度并不是作为一个特征。

你了解Local Outlier Factor吗？我想要：

```
在计算出离群值（例如使用局部离群因子 LOF）之后，需要设置一个门限（阈值）来区分异常点和正常点。门限的设置会直接影响异常检测的结果，选择合适的门限能够更准确地区分异常点和正常点。以下是几种常见的设置门限的方法：

1. 基于统计的方法设置门限
百分位数法（Percentile Method）：使用 LOF 分数的百分位数来设置阈值。例如，可以选择 LOF 分数的 90% 或 95% 作为阈值，这样可以识别最异常的 10% 或 5% 的点。

步骤：
计算所有点的 LOF 分数。
确定一个百分位数（如 90% 或 95%）。
将百分位数的 LOF 值作为阈值，大于该阈值的点被标记为异常点。
均值和标准差法（Mean and Standard Deviation Method）：计算 LOF 分数的均值和标准差，设置为 Mean + k * Standard Deviation 的形式作为阈值，k 是一个超参数，通常取 2 或 3。

步骤：
计算 LOF 分数的平均值和标准差。
使用公式 threshold = Mean + k * Standard Deviation 计算阈值。
2. 基于混合方法调整的门限
在将置信度调整到 LOF 分数后，可以使用上面的方法来设置阈值。以下是如何将置信度与 LOF 结合的一个示例步骤：

计算初始 LOF 分数：使用 LOF 算法计算每个点的初始 LOF 分数。

使用置信度调整 LOF 分数：假设置信度越高，LOF 分数应该越低，可以用置信度的倒数来调整 LOF 分数：

adjusted_lof
=
LOF分数
×
(
1
/
置信度
)
adjusted_lof=LOF分数×(1/置信度)
设置阈值：根据调整后的 LOF 分数，使用百分位数法或均值和标准差法设置门限。
```





尽量不要设置一个固定的值，例如“95%”，这在论文中不应出现。关于这个值应该有个更加理想化的描述。






对于`选择一个适当的百分位数作为阈值`，有没有一种更加具有算法描述的方法








参考之前的描述，给出更多的公式





将` k-dist ( 𝑥 𝑗 ) k-dist(x j ​ ) 表示点 𝑥 𝑗 x j ​ 到其第 𝑘 k 个最近邻的距离`这里的k换成\varepsilon





参考之前的论文格式，将这部分写成论文




我只需要一种方法：调整后的均值和标准差法。不需要“百分位数法”。

同时，不是假设置信度C_i已经计算了出来，而是说根据“最小覆盖球”得出的半径来计算。






<!-- 很棒，写完“调整后的均值和标准差法”还需要再写一步：根据标准差来计算 -->

很棒，`其中，𝑘k 是一个超参数，通常根据数据特性和实验结果来优化。大于这个阈值的点将被标记为异常点。`，这个$k$有没有一种更加学术化的确定方式？“通常根据数据特性和实验结果来优化”写到论文里会让人感觉很不专业





很棒，请将改进后的描述融入到这次的整体描述中






很棒，参考之前的论文格式，返回这部分的latex源码
```
\item \textbf{Single-round Malicious Node Detection}

最后Single-round Malicious Node Detection的的主要过程如算法\ref{alg:malicious-node-detection}的28-29行所示。利用HDBSCAN对计算出来的KL散度集合$S$进行聚类，得到簇\(C_1, C_2, \ldots, C_K\)，从所有簇中选出最大的一簇（即包含节点数量最多的簇），记作 \(C_{max}\)。基于每轮参与训练的良性节点相较于恶意节点数量较多的假设，并且相较于恶意节点，所有良性节点与干净分量之间的KL散度具有更高的相似性，因此将 \(C_{max}\)中的所有节点标记为良性节点， 记作为\(U_{nor}\)。对于除\(C_{max}\)外的所有其他簇\(C_i\)（\(i \neq {max}\)），将其中所有节点标记为恶意节点\(U_{mal}\)。 

\end{enumerate}

\subsection{多轮次攻击识别}

\textbf{Motivation}: 后门攻击

\textbf{Overview}: 前面的攻击中已经可以有效地针对单轮次攻击进行识别，本部分主要结合客户端的历史记录对客户端进行有效的识别。本部分将保留历史$T$轮次的全局模型$G_t$，并将其Flatten后视为高维空间中的一个点；同时保留每个客户端的历史梯度变化，将其视为高维空间中的一个向量，将其延伸后视为一个射线。对于每个客户端，使用最小覆盖球算法计算得到一个最小的超球，覆盖掉$\varepsilon$比例的射线。最终球心的位置可以被视为对应客户端对全局模型的意图引导点，球心的半径可以视为对应客户端的置信度。

\begin{algorithm}
\caption{Malicious Node Detection via History}
\label{alg:malicious-node-detection-history}
\begin{algorithmic}[1]
\State \textbf{Input:} TODO:
\State Step 1: Keep T-rounds History
\For {Node $l_i$}
    \State $\bar{\theta_i^t} \gets Flatten(\triangledown \theta_{l_i}^t)$
    \State $DB.gradient\ \ +=\ \ \bar{\theta_i^t}$
    \State $DB.model\ \ +=\ \ G_{t-1}$
\EndFor
\end{algorithmic}
\end{algorithm}

算法\ref{alg:malicious-node-detection-history}展示了根据每个客户端历史记录进行识别的识别过程，主要包含TODO:个步骤：

\begin{enumerate}

\item \textbf{Keep T-rounds History}

“Keep T-rounds History”的主要算法过程如算法\ref{alg:malicious-node-detection-history}的TODO:行所示。像正常的联邦学习过程一样，中央服务器每次下发一个全局模型，每个客户端得到全局模型后使用本地数据进行训练，并将梯度变化上传到中央服务器中。我们将第$t$轮次的全局模型记为$G_t$，将客户端$l_i$第$t$轮次训练后的模型记为$\theta^t_{l_i}$。客户端$l_i$将训练后的模型$\theta^t_{l_i}$减去训练前中央服务器下发的全局模型$G_t$，就得到了$t$轮次的梯度变化$\triangledown \theta^t_{l_i}=\theta^t_{l_i}-G_{t-1}$。客户端将梯度变化$\triangledown \theta^t_{l_i}$上传到中央服务器，中央服务器在聚合的同时，记录下每个客户端当前轮次的梯度变化$\triangledown \theta^t_{l_i}$展平后的结果$\bar{\theta_i^t}$，同时存下中央服务器上次下发的全局模型$G_{t-1}$。中央服务器最多保留$T$轮次的模型和梯度历史记录。此部分可以参考算法\ref{alg:malicious-node-detection-history}的step1。

\item \textbf{Obtain the Purpose Intention}

“Obtain the Purpose Intention”的主要算法过程如算法\ref{alg:malicious-node-detection-history}的TODO:行所示。这个问题可以抽象为高维空间中的一些具有起点的射线。射线的起点代表$t-1$轮次的全局模型，射线的方向代表$t$轮次展平后的梯度变化。问题的优化目标是：找到一个最小的超球，包含至少$\varepsilon$比例的射线。此部分又可以分为“Construct Ray Model”和“Minimum Enclosing Hypersphere Calculation”两部分。

“Construct Ray Model”的主要过程如下。对于每个客户端 \(l_i\)，中央服务器保留了最近 \(T\) 轮的全局模型历史以及每轮次的梯度更新历史。

\begin{itemize}
    \item 设 \( G_{t-T}, G_{t-T+1}, \dots, G_{t-1} \) 是最近 \(T\) 轮次的全局模型，且这些全局模被“展平”到了高维空间中的一个点。类似地，设 \( \overline{\theta_{l_i}^{t-T}}, \overline{\theta_{l_i}^{t-T+1}}, \dots, \overline{\theta_{l_i}^{t-1}} \) 是最近 \(T\) 轮次客户端 \(l_i\) 上传的展平后的梯度变化向量。
    \item 对于每一轮 \( t-j \) (\( j = 0, 1, \dots, T-1 \))，我们将每个射线的起点表示为展平后的全局模型 \( G_{t-j-1} \)，并将每个射线的方向向量表示为展平后的梯度变化 \( \overline{\theta}_{l_i}^{t-j} \)。因此，射线模型可以构建如下：

    \begin{equation}
    \overline{\theta_{l_i}^{t-j}} = Flatten(G_{t-j-1}),
    \end{equation}

    \begin{equation}
    \mathbf{d}_{l_i}^{t-j} = \bar{\theta}_{l_i}^{t-j},
    \end{equation}

    其中，\( \overline{\theta_{l_i}^{t-j}} \) 是射线的起点，\( \mathbf{d}_{l_i}^{t-j} \) 是射线的方向向量。
\end{itemize}

“Minimum Enclosing Hypersphere Calculation”的主要过程如下。基于构建的射线模型，使用最小覆盖球算法找到一个能够覆盖至少 \( \varepsilon \) 比例的射线的最小超球。

\begin{itemize}
    \item 首先，设定初始球心 \( O_0 \) 为所有射线起点的几何中心，计算公式如下：

    \begin{equation}
    O_0 = \frac{1}{T} \sum_{j=0}^{T-1} \overline{\theta_{l_i}^{t-T+1}}.
    \end{equation}

    \item 初始半径 \( r_0 \) 设定为从初始球心 \( O_0 \) 到所有射线起点的最大距离：

    \begin{equation}
    r_0 = \max_{1\leq T} \|O_0 - \overline{\theta_{l_i}^{t-T+1}}\|.
    \end{equation}

    \item 接下来，采用迭代方法来更新球心和半径：
    \begin{itemize}
       \item 对于每次迭代 \( k \)，计算当前球心 \( O_k \) 到所有射线的最近投影点 \( Q_{l_i}^{t-j} \)，按照距离当前球心的位置排序

        \begin{equation}
        \text{Sort}(\{ \| O - \overline{\theta_{l_i}^{t-T+1}} \| : i = 1, 2, \ldots, T \})
        \end{equation}

        并计算这些投影点的最大距离 \( D_k \)：

        \begin{equation}
        D_k = \max_{1\leq j\leq \lceil n\times \varepsilon\rceil} \|O_k - Q_{l_i}^{t-j}\|.
        \end{equation}

        \begin{equation}
        O_{k+1} = O_k + \alpha \left(\frac{1}{\lceil n\times \varepsilon\rceil} \sum_{1\leq j\leq \lceil n\times \varepsilon\rceil} (Q_{l_i}^{t-j} - O_k)\right),
        \end{equation}

        \begin{equation}
        r_{k+1} = D_k,
        \end{equation}

        其中，\( \alpha \) 是学习率，通常取一个小的正数（如 0.1）。
    \end{itemize}
    \item 终止条件：当更新后的半径 \( r_{k+1} \) 与上一次的半径 \( r_k \) 相差小于一个预设的阈值 \( \epsilon \)（如 \( 10^{-6} \)），或者达到预设的最大迭代次数时，停止迭代。
\end{itemize}

这部分可以定义为一个有约束的最优化问题：

\[
\min_{O, r} \, r 
\]

\[
\text{s.t.} \quad \| O - (\overline{\theta_{l_i}^{t-T+1}} + t_i \mathbf{d}_i) \|^2 \leq r^2, \, 1\leq i\leq \lceil n \times \varepsilon \rceil
\]

\item \textbf{Abnormal Detection}

\item \textbf{Gradients Aggregation}

\end{enumerate}
```






下面写梯度聚合部分，这部分使用主观逻辑模型聚合客户端梯度。

大致流程是：“在上一步已经剔除了恶意客户端”，这一步使用上一步得到的结果，使用主观逻辑模型进行梯度聚合。先返回中文描述。





给出对应的latex源码










根据当前的latex，补全伪代码
```
\subsection{多轮次攻击识别}

\textbf{Motivation}: 后门攻击

\textbf{Overview}: 前面的攻击中已经可以有效地针对单轮次攻击进行识别，本部分主要结合客户端的历史记录对客户端进行有效的识别。本部分将保留历史$T$轮次的全局模型$G_t$，并将其Flatten后视为高维空间中的一个点；同时保留每个客户端的历史梯度变化，将其视为高维空间中的一个向量，将其延伸后视为一个射线。对于每个客户端，使用最小覆盖球算法计算得到一个最小的超球，覆盖掉$\varepsilon$比例的射线。最终球心的位置可以被视为对应客户端对全局模型的意图引导点，球心的半径可以视为对应客户端的置信度。

\begin{algorithm}
\caption{Malicious Node Detection via History}
\label{alg:malicious-node-detection-history}
\begin{algorithmic}[1]
\State \textbf{Input:} TODO:   这里要补全
\State Step 1: Keep T-rounds History
\For {Node $l_i$}
    \State $\bar{\theta_i^t} \gets Flatten(\triangledown \theta_{l_i}^t)$
    \State $DB.gradient\ \ +=\ \ \bar{\theta_i^t}$
    \State $DB.model\ \ +=\ \ G_{t-1}$
\EndFor
\end{algorithmic}
\end{algorithm}

算法\ref{alg:malicious-node-detection-history}展示了根据每个客户端历史记录进行识别的识别过程，主要包含TODO:个步骤：

\begin{enumerate}

\item \textbf{Keep T-rounds History}

“Keep T-rounds History”的主要算法过程如算法\ref{alg:malicious-node-detection-history}的TODO:行所示。像正常的联邦学习过程一样，中央服务器每次下发一个全局模型，每个客户端得到全局模型后使用本地数据进行训练，并将梯度变化上传到中央服务器中。我们将第$t$轮次的全局模型记为$G_t$，将客户端$l_i$第$t$轮次训练后的模型记为$\theta^t_{l_i}$。客户端$l_i$将训练后的模型$\theta^t_{l_i}$减去训练前中央服务器下发的全局模型$G_t$，就得到了$t$轮次的梯度变化$\triangledown \theta^t_{l_i}=\theta^t_{l_i}-G_{t-1}$。客户端将梯度变化$\triangledown \theta^t_{l_i}$上传到中央服务器，中央服务器在聚合的同时，记录下每个客户端当前轮次的梯度变化$\triangledown \theta^t_{l_i}$展平后的结果$\bar{\theta_i^t}$，同时存下中央服务器上次下发的全局模型$G_{t-1}$。中央服务器最多保留$T$轮次的模型和梯度历史记录。此部分可以参考算法\ref{alg:malicious-node-detection-history}的step1。

\item \textbf{Obtain the Purpose Intention}

“Obtain the Purpose Intention”的主要算法过程如算法\ref{alg:malicious-node-detection-history}的TODO:行所示。这个问题可以抽象为高维空间中的一些具有起点的射线。射线的起点代表$t-1$轮次的全局模型，射线的方向代表$t$轮次展平后的梯度变化。问题的优化目标是：找到一个最小的超球，包含至少$\varepsilon$比例的射线。此部分又可以分为“Construct Ray Model”和“Minimum Enclosing Hypersphere Calculation”两部分。

“Construct Ray Model”的主要过程如下。对于每个客户端 \(l_i\)，中央服务器保留了最近 \(T\) 轮的全局模型历史以及每轮次的梯度更新历史。

\begin{itemize}
    \item 设 \( G_{t-T}, G_{t-T+1}, \dots, G_{t-1} \) 是最近 \(T\) 轮次的全局模型，且这些全局模被“展平”到了高维空间中的一个点。类似地，设 \( \overline{\theta_{l_i}^{t-T}}, \overline{\theta_{l_i}^{t-T+1}}, \dots, \overline{\theta_{l_i}^{t-1}} \) 是最近 \(T\) 轮次客户端 \(l_i\) 上传的展平后的梯度变化向量。
    \item 对于每一轮 \( t-j \) (\( j = 0, 1, \dots, T-1 \))，我们将每个射线的起点表示为展平后的全局模型 \( G_{t-j-1} \)，并将每个射线的方向向量表示为展平后的梯度变化 \( \overline{\theta}_{l_i}^{t-j} \)。因此，射线模型可以构建如下：

    \begin{equation}
    \overline{\theta_{l_i}^{t-j}} = Flatten(G_{t-j-1}),
    \end{equation}

    \begin{equation}
    \mathbf{d}_{l_i}^{t-j} = \bar{\theta}_{l_i}^{t-j},
    \end{equation}

    其中，\( \overline{\theta_{l_i}^{t-j}} \) 是射线的起点，\( \mathbf{d}_{l_i}^{t-j} \) 是射线的方向向量。
\end{itemize}

“Minimum Enclosing Hypersphere Calculation”的主要过程如下。基于构建的射线模型，使用最小覆盖球算法找到一个能够覆盖至少 \( \varepsilon \) 比例的射线的最小超球。

\begin{itemize}
    \item 首先，设定初始球心 \( O_0 \) 为所有射线起点的几何中心，计算公式如下：

    \begin{equation}
    O_0 = \frac{1}{T} \sum_{j=0}^{T-1} \overline{\theta_{l_i}^{t-T+1}}.
    \end{equation}

    \item 初始半径 \( r_0 \) 设定为从初始球心 \( O_0 \) 到所有射线起点的最大距离：

    \begin{equation}
    r_0 = \max_{1\leq T} \|O_0 - \overline{\theta_{l_i}^{t-T+1}}\|.
    \end{equation}

    \item 接下来，采用迭代方法来更新球心和半径：
    \begin{itemize}
       \item 对于每次迭代 \( k \)，计算当前球心 \( O_k \) 到所有射线的最近投影点 \( Q_{l_i}^{t-j} \)，按照距离当前球心的位置排序

        \begin{equation}
        \text{Sort}(\{ \| O - \overline{\theta_{l_i}^{t-T+1}} \| : i = 1, 2, \ldots, T \})
        \end{equation}

        并计算这些投影点的最大距离 \( D_k \)：

        \begin{equation}
        D_k = \max_{1\leq j\leq \lceil n\times \varepsilon\rceil} \|O_k - Q_{l_i}^{t-j}\|.
        \end{equation}

        \begin{equation}
        O_{k+1} = O_k + \alpha \left(\frac{1}{\lceil n\times \varepsilon\rceil} \sum_{1\leq j\leq \lceil n\times \varepsilon\rceil} (Q_{l_i}^{t-j} - O_k)\right),
        \end{equation}

        \begin{equation}
        r_{k+1} = D_k,
        \end{equation}

        其中，\( \alpha \) 是学习率，通常取一个小的正数（如 0.1）。
    \end{itemize}
    \item 终止条件：当更新后的半径 \( r_{k+1} \) 与上一次的半径 \( r_k \) 相差小于一个预设的阈值 \( \epsilon \)（如 \( 10^{-6} \)），或者达到预设的最大迭代次数时，停止迭代。
\end{itemize}

这部分可以定义为一个有约束的最优化问题：

\[
\min_{O, r} \, r 
\]

\[
\text{s.t.} \quad \| O - (\overline{\theta_{l_i}^{t-T+1}} + t_i \mathbf{d}_i) \|^2 \leq r^2, \, 1\leq i\leq \lceil n \times \varepsilon \rceil
\]

\item \textbf{Abnormal Detection}

“Abnormal Detection”的主要过程基于局部离群因子（Local Outlier Factor, LOF）计算，并结合“最小覆盖球”得出的置信度来调整 LOF 分数。异常检测的目标是识别出异常的客户端，确保联邦学习过程中的全局模型安全性和鲁棒性。

\begin{itemize}
    \item 首先，计算局部离群因子（LOF）。对于一个样本点 \(x_i\)，其局部可达密度（Local Reachability Density, LRD）定义为其相对于 \(\varepsilon\)-邻域内点的平均可达距离的倒数：

    \begin{equation}
    \text{LRD}(x_i) = \frac{1}{\frac{1}{\varepsilon} \sum_{x_j \in \mathcal{N}_\varepsilon(x_i)} \text{reach-dist}_\varepsilon(x_i, x_j)},
    \end{equation}

    其中，\(\text{reach-dist}_\varepsilon(x_i, x_j)\) 是点 \(x_i\) 到其邻居 \(x_j\) 的可达距离，定义为：

    \begin{equation}
    \text{reach-dist}_\varepsilon(x_i, x_j) = \max(\text{dist}(x_i, x_j), \varepsilon\text{-dist}(x_j)),
    \end{equation}

    这里，\(\varepsilon\text{-dist}(x_j)\) 表示点 \(x_j\) 到其第 \(\varepsilon\) 个最近邻的距离。

    \item 对于点 \(x_i\)，其局部离群因子（LOF）定义为其邻域内所有点的局部可达密度与自身局部可达密度的比值的平均值：

    \begin{equation}
    \text{LOF}(x_i) = \frac{1}{\varepsilon} \sum_{x_j \in \mathcal{N}_\varepsilon(x_i)} \frac{\text{LRD}(x_j)}{\text{LRD}(x_i)}.
    \end{equation}

    \item 接下来，基于“最小覆盖球”方法计算得到的置信度来调整 LOF 分数。假设通过“最小覆盖球”算法得到了客户端 \(i\) 的最小覆盖球半径 \(r_i\)，定义置信度 \(C_i\) 为半径的倒数：

    \begin{equation}
    C_i = \frac{1}{r_i + \epsilon},
    \end{equation}

    其中，\(\epsilon\) 是一个很小的正数，用于避免除零错误。

    \item 使用置信度 \(C_i\) 的倒数来调整 LOF 分数，使得置信度越高的点，其调整后的 LOF 分数越低：

    \begin{equation}
    \text{adjusted\_lof}(x_i) = \text{LOF}(x_i) \times (r_i + \epsilon).
    \end{equation}

    \item 计算调整后的 LOF 分数的均值 \(\mu_{\text{adjusted\_lof}}\) 和标准差 \(\sigma_{\text{adjusted\_lof}}\)，然后使用以下公式设置阈值 \(\tau'\)：

    \begin{equation}
    \tau' = \mu_{\text{adjusted\_lof}} + k \times \sigma_{\text{adjusted\_lof}},
    \end{equation}

    其中，\(k\) 是一个\textbf{敏感性系数}（sensitivity coefficient），其选择对异常检测的结果具有重要影响。为了在精确度与召回率之间取得平衡，\(k\) 的值应通过\textbf{基于数据驱动的交叉验证策略}来确定。具体而言，可以在训练集中使用不同的 \(k\) 值进行多次迭代，计算每种情况下的检测性能指标（如精确度、召回率和 F1 分数），并通过优化这些指标选择最优的 \(k\) 值。此外，选择 \(k\) 时也可以考虑使用贝叶斯优化（Bayesian Optimization）等先进的超参数调优方法，以自适应地选择最能反映数据特征的敏感性系数值。对于检测结果中大于该阈值的点，将其判定为异常点。
\end{itemize}


\item \textbf{Gradients Aggregation}

在上一步已经剔除了恶意客户端后，本部分将使用主观逻辑模型（Subjective Logic Model）进行剩余客户端的梯度聚合。主观逻辑模型是一种基于概率论和信任度的数学框架，能够有效地处理不确定性和不一致性信息。在联邦学习场景中，主观逻辑模型可以用于在不完全信任的环境下对客户端上传的梯度进行加权聚合，从而提高全局模型的稳健性。

\begin{itemize}
    \item \textbf{定义主观逻辑模型的信任参数}：

    对于每个被认为是正常的客户端 \( C_i \)，根据其上传的梯度更新 \( \triangledown \theta_i \)，定义主观逻辑模型中的信任参数，包括\textbf{信任度}（Belief）、\textbf{不信任度}（Disbelief）和\textbf{不确定度}（Uncertainty）。信任度 \( b_i \) 可以根据每个客户端的历史表现或其他统计指标来定义。对于本次的应用，信任度可以与上一步异常检测中调整后的 LOF 分数成反比。

    \item \textbf{计算每个客户端的证据权重}：

    基于定义的信任参数，为每个客户端 \( C_i \) 计算其证据权重 \( w_i \)。证据权重可以表示为信任度 \( b_i \) 的函数形式，例如：\( w_i = f(b_i) \)，其中 \( f(b_i) \) 可以是线性或非线性的映射函数。

    \item \textbf{加权聚合客户端梯度}：

    使用证据权重 \( w_i \) 对所有正常客户端的梯度更新 \( \triangledown \theta_i \) 进行加权聚合，以得到全局模型的梯度更新：

    \begin{equation}
    \triangledown \theta_{\text{global}} = \sum_{i \in U_{nor}} w_i \cdot \triangledown \theta_i
    \end{equation}

    其中，\( U_{nor} \) 是所有被判定为正常的客户端集合。

    \item \textbf{更新全局模型}：

    使用聚合后的梯度更新全局模型参数，以继续下一轮的联邦学习迭代。
\end{itemize}

通过使用主观逻辑模型对客户端梯度进行聚合，能够有效利用信任度信息来优化全局模型的更新过程，减少因恶意客户端或不确定性导致的模型偏差。

\end{enumerate}
```
你只需要返回`algorithm`部分的latex源码即可






要求全英文描述







论文的“实验设置”部分应该怎么写




我正在写一篇论文，请你帮我写“实验设置”的“实验环境”部分。

其中我的配置如下：
```
% 本实验在一台配置高性能硬件的Ubuntu 20.04.3 LTS操作系统的计算机上进行，具体配置如下：

% 操作系统：Ubuntu 20.04.3 LTS，内核版本 5.15.0-73-generic
% CPU：Intel(R) Core(TM) i9-10940X CPU @ 3.30GHz，28核，最大频率 4.8GHz
% 内存：128GB DDR4，2933 MT/s
% GPU：2 x NVIDIA GeForce RTX 3090，显存各 24GB，驱动版本 470.239.06，CUDA 版本 11.4
% 存储：SSD，提供高速数据读写支持
% 为了确保实验结果的可靠性和再现性，我们使用了标准的数据集，包括MNIST数据集和一个大型工业产品推荐数据集。实验中使用的具体硬件配置能够提供充足的计算资源，支持大规模深度学习模型的训练和复杂算法的测试。
```




返回一段latex




写成一长段话






我准备使用的数据集为





好的谢谢！






Python模拟按键，先往Chrome浏览器窗口输入快捷键`Ctrl+A`、`Ctrl+V`，再往微信窗口输入`Enter`。





能不能定时然后把鼠标移动到指定的位置





写段代码，我鼠标单击左键后，返回鼠标位置






Chrome窗口要点击的位置是`Point(x=383, y=291)`，微信窗口要点击的位置是`Point(x=1184, y=748)`





定时，凌晨3点执行





只需要执行一次，sleep到3点02开始执行






进行以下修改：
1. Chrome浏览器窗口输入快捷键`Ctrl+A`、`Ctrl+V`
2. 往微信窗口输入`Enter`
3. 全部执行完毕后，几秒钟后执行快捷键`Win+L`






`pyautogui.hotkey('win', 'l')`这一步执行失败