<!--
 * @Author: LetMeFly
 * @Date: 2024-08-18 10:06:39
 * @LastEditors: LetMeFly
 * @LastEditTime: 2024-09-15 02:00:25
-->




简写这段话，只描述一些主要的配置
```
本实验在一台配置高性能硬件的计算机上进行，操作系统为 Ubuntu 20.04.3 LTS，内核版本为 5.15.0-73-generic，以确保计算稳定性和结果的可靠性。计算机配置了 Intel(R) Core(TM) i9-10940X @ 3.30GHz 的处理器，具有28个核心，最大频率可达4.8GHz，能够提供强大的并行计算能力以高效处理大规模计算任务。内存为 128GB DDR4，速度为 2933 MT/s，这种大容量的内存配置有助于在处理复杂数据集和大规模模型训练时减少内存瓶颈，提升数据加载和处理速度。图形处理单元（GPU）方面，采用了2块 NVIDIA GeForce RTX 3090，每块显存为24GB，驱动版本为 470.239.06，CUDA 版本为 11.4。双GPU配置提供了强大的并行计算能力，可以显著加速深度学习模型的训练过程，特别是在图像处理和自然语言处理等需要大量计算的任务中表现尤为出色。存储设备为高速SSD，支持数据的高速读写操作，显著降低I/O操作时间，确保大数据集的快速加载和模型的高效训练。为了确保实验结果的可靠性和再现性，本实验使用了标准的数据集，包括MNIST数据集、CIFAR-10数据集等主流的数据集。上述硬件配置为实验提供了充足的计算资源，能够支持大规模深度学习模型的训练和复杂算法的测试。我们采用了一些先进的攻击方式，并使用我们的防御方式和State-Of-The-Art以及较为主流的防御方式进行对比，并针对“结合多轮次进行攻击”的攻击方式进行了实验，验证了我们的有效性。
```




```
\subsection{EXPERIMENTAL EVALUATION}

\subsubsection{Experimental Setup}

\subsubsection{硬件设施}

本实验在一台Ubuntu操作系统的服务器上进行，28核心128GB内存，以及2块NVIDIA GeForce RTX 3090 GPU。

\subsubsection{基本设置}

\subsubsection{攻防相关}
```
现在写“基本设置”部分，包括：
```
数据集 模型  联邦相关机制（聚合算法）  一定有引用
参数比如学习率 本地轮次 神经元之类的

甚至列表

参数取值表、
```
返回中文版描述及Latex源码





模型使用的是ViT






CIFAR-10如何引用






解释这段代码
```
def build_cache_model(cfg, clip_model: CLIP, train_loader_cache: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:

    if cfg['load_cache'] == False:    
        cache_keys = []
        cache_values = []

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(cfg['augment_epoch']):
                train_features = []

                print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
                for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                    images = images.cuda()
                    image_features = clip_model.encode_image(images)
                    train_features.append(image_features)
                    if augment_idx == 0:
                        target = target.cuda()
                        cache_values.append(target)
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))
            
        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0)
        cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()

        torch.save(cache_keys, cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        torch.save(cache_values, cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")

    else:
        cache_keys: torch.Tensor = torch.load(cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        cache_values: torch.Tensor = torch.load(cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")

    return cache_keys, cache_values
```





CLIP模型是怎么训练的





换一种不共享内存的方法





介绍主观逻辑模型






结合公式介绍主观逻辑模型





信任度三元组 的初始值





详细介绍 概率期望值






联邦学习中，每个客户端有一些主观逻辑模型评分，我在聚合客户端梯度的时候如何据此进行加权





如果一个客户端被评为恶意客户端的话，一般是直接不聚合它的梯度，还是按照主观逻辑模型计算出来的结果进行聚合呢







联邦学习中，我计算出了每个客户端的意图点，使用一个圆来表示。圆的半径越小则表示这个客户端的置信度越高，在聚合的时候权重就应该越大。我想应该如何依据半径来确定聚合时的权重？





我能否使用类似激活函数的方式来计算权重？因为半径较小时（0.0001和0.00000001）其实都表示置信度很高，权重应该差不多。但其实它们相差了很多倍。









```
\subsection{多轮次攻击识别}

\textbf{Motivation}: 为了规避异常检测，有很多通过多轮次的组合攻击来植入后门的攻击。本部分结合每个客户端的历史记录，通过识别攻击者最终的模型引导意图点来识别恶意节点。

\textbf{Overview}: 前面的攻击中已经可以有效地针对单轮次攻击进行识别，本部分主要结合客户端的历史记录对客户端进行有效的识别。本部分将保留历史$T$轮次的全局模型$\theta_g^{t'}$，其中$\max\{t-T+1, 1\}\leq  t'\leq t$，并将其Flatten后视为高维空间中的一个点；同时保留每个客户端的历史更新变化，将其视为高维空间中的一个向量，将其延伸后视为一个射线。对于每个客户端，使用最小覆盖球算法计算得到一个最小的超球，覆盖$\zeta$比例的射线。最终球心的位置可以被视为对应客户端对全局模型的意图引导点，球心的半径可以视为对应客户端的置信度。

\begin{algorithm}
\caption{Malicious Node Detection via History}
\label{alg:malicious-node-detection-history}
\begin{algorithmic}[1]
\State \textbf{Input:} Global model history $\{G_{t-T}, G_{t-T+1}, \dots, G_{t-1}\}$, historical gradient updates for each client $\{\triangledown \theta_{i}^{t-T}, \triangledown \theta_{i}^{t-T+1}, \dots, \triangledown \theta_{i}^{t-1}\}$, threshold $\varepsilon$
\State \textbf{Output:} Set of clients marked as normal $U_{nor}$ and set of clients marked as malicious $U_{mal}$
\State Step 1: Keep T-rounds History
\For {Node $i$}
    \State $\bar{\theta_i^t} \gets Flatten(\triangledown \theta_{i}^t)$
    \State $DB.gradient\ \ +=\ \ \bar{\theta_i^t}$
    \State $DB.model\ \ +=\ \ G_{t-1}$
\EndFor
\State Step 2: Construct Ray Model
\For {Node $i$}
    \State Construct rays using flattened global models and gradient updates:
    \State $P_{i}^{t-j} = Flatten(G_{t-j-1})$, $\mathbf{d}_{i}^{t-j} = \bar{\theta}_{i}^{t-j}$ for $j = 0, 1, \ldots, T-1$
\EndFor
\State Step 3: Minimum Enclosing Hypersphere Calculation
\For {Node $i$}
    \State Initialize center $O_0$ and radius $r_0$
    \While {Not Converged}
        \State Calculate projection points $Q_{i}^{t-j}$ and sort by distance to $O_k$
        \State Compute $D_k = \max_{1\leq j\leq \lceil n\times \varepsilon\rceil} \|O_k - Q_{i}^{t-j}\|$
        \State Update $O_{k+1} = O_k + \alpha \left(\frac{1}{\lceil n\times \varepsilon\rceil} \sum_{1\leq j\leq \lceil n\times \varepsilon\rceil} (Q_{i}^{t-j} - O_k)\right)$
        \State Update $r_{k+1} = D_k$
    \EndWhile
    \State Calculate confidence $C_i = \frac{1}{r_i + \epsilon}$
\EndFor
\State Step 4: Abnormal Detection
\For {Node $i$}
    \State Calculate adjusted LOF: $\text{adjusted\_lof}(x_i) = \text{LOF}(x_i) \times (r_i + \epsilon)$
    \State Determine threshold $\tau'$ and classify nodes as normal or malicious
\EndFor
\State \textbf{Return} $U_{nor}, U_{mal}$
\end{algorithmic}
\end{algorithm}

算法\ref{alg:malicious-node-detection-history}展示了根据每个客户端历史记录进行识别的识别过程，主要包含4个步骤：

\begin{enumerate}

\item \textbf{Keep T-rounds History}

“Keep T-rounds History”的主要算法过程如算法\ref{alg:malicious-node-detection-history}的3-13行所示。像正常的联邦学习过程一样，中央服务器每次下发一个全局模型，每个客户端得到全局模型后使用本地数据进行训练，并将更新变化上传到中央服务器中。我们将第$t$轮次的全局模型记为$\theta_t$，将客户端$i$第$t$轮次训练后的模型记为$\theta^t_i$。客户端$i$将训练后的模型$\theta^t_{i}$减去训练前中央服务器下发的全局模型$\theta^t$，就得到了$t$轮次的更新变化$\triangledown \theta^t_{i}=\theta^t_{i}-\theta^t$。客户端将更新变化$\triangledown \theta^t_{i}$上传到中央服务器，中央服务器在聚合的同时，记录下每个客户端当前轮次的更新$\triangledown \theta^t_{i}$展平后的结果$\bar{\theta_i^t}$，同时存下中央服务器上轮次下发的全局模型$\theta^{t-1}$。中央服务器最多保留$T$轮次的模型和更新历史记录。此部分可以参考算法\ref{alg:malicious-node-detection-history}的step1。

\item \textbf{Obtain the Purpose Intention}

“Obtain the Purpose Intention”的主要算法过程如算法\ref{alg:malicious-node-detection-history}的14-18行所示。这个问题可以抽象为高维空间中的一些具有起点的射线。射线的起点代表上轮次的全局模型，射线的方向代表本轮次展平后的更新变化。问题的优化目标是：找到一个最小的超球，至少覆盖$\zeta$比例的射线。此部分又可以分为“Construct Ray Model”和“Minimum Enclosing Hypersphere Calculation”两部分。

“Construct Ray Model”的主要过程如下。对于每个客户端 \(R_i\)，中央服务器保留了最近 \(T\) 轮的全局模型历史以及每轮次的更新变化历史。设 \( \theta_g^{t-T}, \theta_g^{t-T+1}, \dots, \theta_g^{t-1} \) 是最近 \(T\) 轮次的全局模型，且这些全局模被“展平”到了高维空间中的一个点。类似地，设 \( \overline{\triangledown\theta_{i}^{t-T}}, \overline{\triangledown\theta_{i}^{t-T+1}}, \dots, \overline{\triangledown\theta_{i}^{t-1}} \) 是最近 \(T\) 轮次客户端 \(i\) 上传的展平后的梯度变化向量。
    对于每一轮 \( t' \) (\( \max\{t-T+1, 1\}\leq  t'\leq t \))，我们将每个射线的起点表示为展平后的全局模型 \( \theta_g^{t'} \)，并将每个射线的方向向量表示为展平后的梯度变化 \( \overline{\triangledown\theta_{i}^{t'}} \)。因此，射线模型可以构建如下：

    \begin{equation}
    \widetilde{\theta_{i}^{t'-1}} = Flatten(\theta_g^{t'-1}),
    \end{equation}
    \begin{equation}
    {v}_{i}^{t'} = \overline{\triangledown\theta_{i}^{t'}},
    \end{equation}

    其中，\( \widetilde{\theta_{i}^{t'-1}} \) 是射线的起点，\( {v}_{i}^{t'} \) 是射线的方向向量。则射线方程为：

    \begin{equation}
    l_i^{t'}=\widetilde{\theta_{i}^{t'-1}}+\alpha {v}_{i}^{t'},
    \end{equation}

    其中，$\alpha\in [0, +\infty)$为射线的自变量参数。


“Minimum Enclosing Hypersphere Calculation”的主要过程如下。基于构建的射线模型，使用最小覆盖球算法找到一个能够覆盖至少 \( \zeta \) 比例的射线的最小超球。首先，设定初始球心 \( O_{i,0} \) 为所有射线起点的几何中心，计算公式如下：

\begin{equation}
    O_{i,0} = \frac{1}{T} \sum_{t'=\max\{t-T+1, 1\}}^{t} \widetilde{\theta_g^{t'-1}}.
\end{equation}

初始半径 \( r_{i,0} \) 设定为从初始球心 \( O_{i,0} \) 到所有射线起点的最大距离：

\begin{equation}
r_{i,0} = \max_{t'=\max\{t-T+1, 1\}}^{t}  \|O_{i,0} - \widetilde{\theta_{g}^{t'-1}}\|.
\end{equation}

接下来，采用迭代方法来更新球心和半径：

对于每次迭代 \( k \)，计算当前球心 \( O_{i,k} \) 到所有射线的最近点 \( \tilde{O}_{i,k}^{t'} \)。对于每个节点$R_i$的第$t'(\max\{t-T+1, 1\}\leq  t'\leq t)$轮，对于射线 \( l_i^{t'} \)，找到球心 \( O_{i,k} \) 与当前射线的投影点 \( \hat O_{i,k}^{t'} \)，并计算参数 \( \alpha_k \)：


    \begin{equation}
    \alpha_k = \frac{(O_{i,k} - \widetilde{\theta_{i}^{t'-1}}) \cdot {v}_{i}^{t'}}{{v}_{i}^{t'}\cdot {v}_{i}^{t'}},
    \end{equation}
    则最近点 \( \tilde O_{i,k}^{t'} =\widetilde{\theta_{i}^{t'-1}}+\max \left\{ 0, \alpha_k\right\} {v}_{i}^{t'} \)。
    
    计算当前球心 \( O_{i,k} \) 到这些最近点 \( \tilde O_{i,k}^{t'} \) 的距离集合 \( Dis_{i,k}= \{ \| O_{i,k}-\tilde O_{i,k}^{t'} \|, \max\{t-T+1, 1\}\leq  t'\leq t\} \)，并按从小到大的顺序进行排序，然后暂时舍弃$1-\zeta$比例的射线来忽略离群点对计算过程的影响，得到集合${Dis}_{i,k}'$，保留射线对应轮次的集合记为$\tilde T$。将球心向着被保留的最近点中距离最远的点$\tilde{O}_{i,k}^{t_{max}'}$移动：
    % ，$ {Dis}_{i,k}'=\text{Sort}(Dis_{i,k})$。
        % 并计算这些投影点的最大距离 \( D_k \)：

        % \begin{equation}
        % D_k = \max_{1\leq j\leq \lceil n\times \varepsilon\rceil} \|O_k - Q_{i}^{t-j}\|.
        % \end{equation}

    \begin{equation}
    O_{i,k+1} = O_{i,k} + \eta' (\tilde{O}_{i,k}^{t_{max}'}-O_{i,k}),
    \end{equation}

    其中$\eta'$为学习率。移动后的球心半径为：

        \begin{equation}
        r_{i,k+1} =  \{ \max\{\| O_{i,k+1}-\tilde O_{i,k+1}^{t'} \|\}, t'\in \tilde T\} ,
        \end{equation}

    其中$\tilde O_{i,k+1}^{t'}$为更新后的球心$O_{i,k+1}$到被保留射线的最近点。终止条件：当更新后的半径 \( r_{i,k+1} \) 与上一次的半径 \( r_{i,k} \) 相差小于一个预设的阈值 \( \lambda \)，或者达到预设的最大迭代次数$k_{max}$时，停止迭代。这部分可以定义为一个有约束的最优化问题：


\begin{equation}
\min_{O_i, r_i} \quad r_i
\end{equation}

\begin{equation} 
\text{s.t.} \quad \left| \left\{ t' \mid \| O_i - \tilde{O}_{i}^{t'} \| \leq r_i \right\} \right| \geq \zeta T.
\end{equation}
% \begin{equation} 
% \quad |r_{i,k+1} - r_{i,k}| < \lambda \quad \text{or} \quad k \geq k_{max}
% \end{equation}

% \textbf{约束条件:}

% 对于每个射线 $l_i^{t'}$，计算球心 $O$ 到射线的最近点 $\tilde{O}_{i}^{t'}$ 的距离。最近点的计算公式为:

% \[
% \tilde{O}_{i}^{t'} = \widetilde{\theta_{i}^{t'-1}} + \max \left\{ 0, \frac{(O - \overline{\theta_{i}^{t'-1}}) \cdot {v}_{i}^{t'}}{{v}_{i}^{t'} \cdot {v}_{i}^{t'}} \right\} {v}_{i}^{t'}
% \]

% 约束球的半径 $r$ 使其能够覆盖至少 $\zeta$ 比例的射线:


% \textbf{迭代更新规则:}

% \[
% O_{k+1} = O_k + \eta' (\tilde{O}_{i,k}^{t_M'} - O_k), \quad r_{k+1} = \max_{t' \in \tilde T} \| O_{k+1} - \tilde{O}_{i,k+1}^{t'} \|
% \]

% 其中 $\tilde T$ 是保留的射线集合，满足覆盖至少 $\zeta$ 比例的射线，$\eta'$ 为学习率。

% \textbf{终止条件:}

% \[
% \text{s.t.} \quad |r_{k+1} - r_k| < \lambda \quad \text{or} \quad k \geq \text{MaxIter},
% \]

% 其中 $\lambda$ 为预设的收敛阈值，$\text{MaxIter}$ 是最大迭代次数。

% \begin{equation}
% \min_{O_i, r_i} \, r_i 
% \end{equation}
% $$
% \text{s.t.} \quad \| O_i - \tilde O_{i}^{t'}\|^2\leq r_i^2 \,
% $$
% $$
% 1\leq t'\leq \lceil n \times \varepsilon \rceil
% $$

\item \textbf{Abnormal Detection}

“Abnormal Detection”的主要过程基于局部离群因子（Local Outlier Factor, LOF）得到异常点。LOF算法是一种基于密度的异常检测方法，算法的核心思想是通过计算一个数值$lof_i$来反应节点$i$的异常程度。$lof_i$大致意思是节点$i$目的意图点$O_i$周围意图点所处位置的平均密度比上$O_i$所在位置的密度。比值越大于$1$，则$O_i$所在位置的密度越小于周围其他意图点所在位置的密度，即$O_i$越有可能是异常点。首先定义意图点$O_i$的$\kappa$距离$\widetilde{dis_i^\kappa}$，假设高维空间中存在节点$j$的意图点$O_j$，则有$O_j$与$O_i$之间的距离为$\|O_i-O_j\|$，如果满足以下两个条件，我们就认为$\widetilde{dis_i^\kappa}=\|O_i-O_j\|$：

\begin{itemize}
    \item 在样本空间中，至少存在$\kappa$个意图点$O_j'$，使得$\|O_i-O_j'\|\leq \|O_i-O_j\|$, 其中$j'\neq i$；
    \item 在样本空间中，至多存在$\kappa-1$个意图点$O_j'$，使得$\|O_i-O_j'\|< \|O_i-O_j\|$, 其中$j'\neq i$。
\end{itemize}

总的来说，$O_i$的$\kappa$距离$\widetilde{dis_i^\kappa}$表示高维空间中距离第$\kappa$远的点。之后定义意图点$O_i$的第$\kappa$距离邻域$Nei_i$为到$O_i$距离不超过$\widetilde{dis_i^\kappa}$的所有意图点的集合。由于可能同时存在多个第$\kappa$距离的数据，所以$|Nei_i|\geq \kappa$。可以想象，离群度越大的意图点的$\kappa$距离往往较大，离群度越小的意图点的$\kappa$距离往往较小。之后定义意图点$O_i$相对于意图点$O_j$的可达距离：

\begin{equation}
rea_{i,j}=\max\{\widetilde{dis_i^\kappa}, \|O_i-O_j\|\},
\end{equation}

也就是说，如果意图点$O_j$远离意图点$O_i$，则两者之间的可达距离就是他们之间的实际距离$\|O_i-O_j\|$；而如果二者距离足够近，则可达距离用意图点$O_i$的$\kappa$距离$\widetilde{dis_i^\kappa}$代替。之后定义意图点$O_i$的局部可达密度$lrd_i$为其$Nei_i$所有意图点的平均可达距离的倒数，即：

\begin{equation}
lrd_i=1/(\frac{\sum_{j\in Nei_i} rea_{i,j}}{|Nei_i|})
\end{equation}

此时，若有重复点，则可能导致$lrd$变为无限大。$lrd_i$的可以理解为意图点$O_i$所处位置的密度，密度越高则意图点$O_i$越有可能属于同一簇，密度越低则意图点$O_i$越有可能是离群点。也就是说，如果意图点$O_i$和周围邻域点是同一簇，则可达距离可能为较小的$\widetilde{dis_i^\kappa}$，导致可达距离之和较小，密度值较高；如果$O_i$和周围邻居意图点较远，则可达距离可能会取较大的$\|O_i-O_j\|$，导致可达距离之和较大，密度值较低，越有可能是离群点。最后，我们定义意图点$O_i$的局部离群因子$lof_i$为其$Nei_i$所有意图点的局部可达密度与其自身局部可达密度的比值的平均值，即：

\begin{equation}
    lof_i=\frac{\sum_{j\in Nei_i}\frac{lrd_j}{lrd_i}}{|Nei_i|}=\frac{\sum_{j\in Nei_i} lrd_j}{|Nei_i|\cdot lrd_i}
\end{equation}

如果$lof_i$比较接近$1$，则说明意图点$O_i$与其邻域点密度差不多，$O_i$可能和邻域属于同一簇；如果$lof_i$小于$1$，则说明意图点$O_i$的密度高于其邻域点的密度，$O_i$为密集点；如果$lof_i$大于$1$，则说明意图点$O_i$的密度低于其邻域点的密度，$O_i$可能是离群点。总之，LOF算法主要通过比较每个意图点$O_i$和其邻域点的密度来判断$O_i$是否为离群点，密度越低，则越有可能是离群点。而密度主要是通过点之间的距离来计算的，点之间的距离越远密度越低，距离越近密度越高。

计算每个节点$i$的局部离群因子$lof_i$，将$lof_i$大于$1$的客户端视为异常客户端，并将其梯度丢弃。

% \begin{itemize}
%     \item 首先，计算局部离群因子（LOF）。对于一个样本点 \(x_i\)，其局部可达密度（Local Reachability Density, LRD）定义为其相对于 \(\varepsilon\)-邻域内点的平均可达距离的倒数：

%     \begin{equation}
%     \text{LRD}(x_i) = \frac{1}{\frac{1}{\varepsilon} \sum_{x_j \in \mathcal{N}_\varepsilon(x_i)} \text{reach-dist}_\varepsilon(x_i, x_j)},
%     \end{equation}

%     其中，\(\text{reach-dist}_\varepsilon(x_i, x_j)\) 是点 \(x_i\) 到其邻居 \(x_j\) 的可达距离，定义为：

%     \begin{equation}
%     \text{reach-dist}_\varepsilon(x_i, x_j) = \max(\text{dist}(x_i, x_j), \varepsilon\text{-dist}(x_j)),
%     \end{equation}

%     这里，\(\eta\varepsilon\text{-dist}(x_j)\) 表示点 \(x_j\) 到其第 \(\varepsilon\) 个最近邻的距离。

%     \item 对于点 \(x_i\)，其局部离群因子（LOF）定义为其邻域内所有点的局部可达密度与自身局部可达密度的比值的平均值：

%     \begin{equation}
%     \text{LOF}(x_i) = \frac{1}{\varepsilon} \sum_{x_j \in \mathcal{N}_\varepsilon(x_i)} \frac{\text{LRD}(x_j)}{\text{LRD}(x_i)}.
%     \end{equation}

%     \item 接下来，基于“最小覆盖球”方法计算得到的置信度来调整 LOF 分数。假设通过“最小覆盖球”算法得到了客户端 \(i\) 的最小覆盖球半径 \(r_i\)，定义置信度 \(C_i\) 为半径的倒数：

%     \begin{equation}
%     C_i = \frac{1}{r_i + \epsilon},
%     \end{equation}

%     其中，\(\epsilon\) 是一个很小的正数，用于避免除零错误。

%     \item 使用置信度 \(C_i\) 的倒数来调整 LOF 分数，使得置信度越高的点，其调整后的 LOF 分数越低：

%     \begin{equation}
%     \text{adjusted\_lof}(x_i) = \text{LOF}(x_i) \times (r_i + \epsilon).
%     \end{equation}

%     \item 计算调整后的 LOF 分数的均值 \(\mu_{\text{adjusted\_lof}}\) 和标准差 \(\sigma_{\text{adjusted\_lof}}\)，然后使用以下公式设置阈值 \(\tau'\)：

%     \begin{equation}
%     \tau' = \mu_{\text{adjusted\_lof}} + k \times \sigma_{\text{adjusted\_lof}},
%     \end{equation}

%     其中，\(k\) 是一个\textbf{敏感性系数}（sensitivity coefficient），其选择对异常检测的结果具有重要影响。为了在精确度与召回率之间取得平衡，\(k\) 的值应通过\textbf{基于数据驱动的交叉验证策略}来确定。具体而言，可以在训练集中使用不同的 \(k\) 值进行多次迭代，计算每种情况下的检测性能指标（如精确度、召回率和 F1 分数），并通过优化这些指标选择最优的 \(k\) 值。此外，选择 \(k\) 时也可以考虑使用贝叶斯优化（Bayesian Optimization）等先进的超参数调优方法，以自适应地选择最能反映数据特征的敏感性系数值。对于检测结果中大于该阈值的点，将其判定为异常点。
% \end{itemize}


\item \textbf{Gradients Aggregation}

\begin{figure}[!t]
    \centering
    \includegraphics[width=\linewidth]{figures/fig5-before-agg.jpg}
    \caption{Benign and Malicious Nodes Before Aggregation}
    \label{fig:before-agg}
\end{figure}

剔除了异常节点后，依据正常节点的置信度$cre$进行聚合。其中定义：

\begin{equation}
    cre_i=\frac{1}{r_i+\rho},
\end{equation}

$\rho$是一个很小的正数，以防分母为$0$。

如图\ref{fig:before-agg}所示，使用LOF算法识别恶意客户端可以剔除潜在的恶意节点，但是在正常的节点中，同样存在一些置信度很低的节点。这些节点的最小覆盖球甚至可能和异常节点有一定交集。因此，在聚合的过程中，这些置信度较低的点所占的权重就应该越低。

试想，假设有两个节点的最小覆盖超球的半径都很小，但是相差倍数很高，那么它们计算出来的置信度相差倍数也会很高。但其实它们的意图点都十分明确，因此权重应该都比较高且相差不应很大。所以我们可以使用激活函数Tanh来对置信度进行加权处理：

\begin{equation}
    cre_i'=\tanh(cre_i),
\end{equation}

激活函数Tanh在$0$到$\infty$范围内是一个上升又快到慢的单调递增函数。当$cre_i\to 0$时$cer_i'\to 0$，也就是说节点$i$的最小覆盖球半径$r_i$很大时该节点在聚合时的权重很小；当$cre_i\to \infty$时$cer_i'\to 1$，也就是说节点$i$的最小覆盖球半径$r_i$很小时该节点在聚合时的权重较大。这样，我们就可以对每个加权处理过的置信度$cer_i'$进行归一化处理，得到权重$w_i$。其中：

\begin{equation}
    w_i=\frac{cre_i'}{\sum_{i=1}^{n}cre_i'},
\end{equation}

因此第$t$轮模型的梯度聚合公式为：

\begin{equation}
    \theta_g^t = \theta_g^{t-1}+\sum_{i=1}^{n}w_i\triangledown \theta_i^t,
\end{equation}

% 在上一步已经剔除了恶意客户端后，本部分将使用主观逻辑模型（Subjective Logic Model）进行剩余客户端的梯度聚合。主观逻辑模型是一种基于概率论和信任度的数学框架，能够有效地处理不确定性和不一致性信息。在联邦学习场景中，主观逻辑模型可以用于在不完全信任的环境下对客户端上传的梯度进行加权聚合，从而提高全局模型的稳健性。其主要流程如算法25-29行所示。

% \begin{itemize}
%     \item \textbf{定义主观逻辑模型的信任参数}：

%     对于每个被认为是正常的客户端 \( C_i \)，根据其上传的梯度更新 \( \triangledown \theta_i \)，定义主观逻辑模型中的信任参数，包括\textbf{信任度}（Belief）、\textbf{不信任度}（Disbelief）和\textbf{不确定度}（Uncertainty）。信任度 \( b_i \) 可以根据每个客户端的历史表现或其他统计指标来定义。对于本次的应用，信任度可以与上一步异常检测中调整后的 LOF 分数成反比。

%     \item \textbf{计算每个客户端的证据权重}：

%     基于定义的信任参数，为每个客户端 \( C_i \) 计算其证据权重 \( w_i \)。证据权重可以表示为信任度 \( b_i \) 的函数形式，例如：\( w_i = f(b_i) \)，其中 \( f(b_i) \) 可以是线性或非线性的映射函数。

%     \item \textbf{加权聚合客户端梯度}：

%     使用证据权重 \( w_i \) 对所有正常客户端的梯度更新 \( \triangledown \theta_i \) 进行加权聚合，以得到全局模型的梯度更新：

%     \begin{equation}
%     \triangledown \theta_{\text{global}} = \sum_{i \in U_{nor}} w_i \cdot \triangledown \theta_i
%     \end{equation}

%     其中，\( U_{nor} \) 是所有被判定为正常的客户端集合。

%     \item \textbf{更新全局模型}：

%     使用聚合后的梯度更新全局模型参数，以继续下一轮的联邦学习迭代。
% \end{itemize}

% 通过使用主观逻辑模型对客户端梯度进行聚合，能够有效利用信任度信息来优化全局模型的更新过程，减少因恶意客户端或不确定性导致的模型偏差。
\end{enumerate}
```

这里面的伪代码是之前的伪代码，现在正文已经更改了，请参考新的正文写一个新的伪代码，并返回其latex源码。






原来的伪代码中的符号没有修改，请使用正文中的符号





1. 代码请简略一点
2. 注意step4中，`adjusted\_lof`、` \text{LOF}(O_i)`这些符号的正确性（正文中并没有出现这些符号，不要使用注释中的内容）




再此基础上修改并补全伪代码的第3和第4步

```
\begin{algorithm}
\caption{Malicious Node Detection and Gradient Aggregation}
\label{alg:malicious-node-detection-history}
\begin{algorithmic}[1]
\State \textbf{Input:} Global model history $\{\theta_g^{t-T}, \dots, \theta_g^{t-1}\}$, historical gradient updates for each client $\{\triangledown \theta_{i}^{t-T}, \dots, \triangledown \theta_{i}^{t-1}\}$, threshold $\varepsilon$
\State \textbf{Output:} Sets of normal clients $U_{nor}$ and malicious clients $U_{mal}$

\State \textbf{Step 1: Keep T-rounds History}
\For {Client $i$}
    \State Record flattened gradients $\bar{\theta_i^t} = Flatten(\triangledown \theta_{i}^t)$ and models $\theta_g^{t-1}$
    \State Construct rays: $\widetilde{\theta_{i}^{t'-1}} = Flatten(\theta_g^{t'-1})$, ${v}_{i}^{t'} = \overline{\triangledown\theta_{i}^{t'}}$ for $t' = \max\{t-T+1, 1\}, \ldots, t$
\EndFor

\State \textbf{Step 2: Obtain the Purpose Intention}
\For {Client $i$}
    \State Initialize center $O_{i,0}$ and radius $r_{i,0}$
    \While {Not Converged}
        \State Update center $O_{i,k+1}$ and radius $r_{i,k+1}$
    \EndWhile
    \State Calculate confidence $cre_i = \frac{1}{r_i + \rho}$
\EndFor

\State \textbf{Step 3: Abnormal Detection}
\State \textbf{Step 4: Gradients Aggregation}
\For {Client $i$ where }
        \State Mark client as normal, add to $U_{nor}$
        \State Compute adjusted confidence $cre_i' = \tanh(cre_i)$
        \State Normalize weights $w_i = \frac{cre_i'}{\sum_{i \in U_{nor}} cre_i'}$
\EndFor

\State \textbf{Output:} Aggregated global model update $\theta_g^t = \theta_g^{t-1} + \sum_{i \in U_{nor}} w_i \cdot \triangledown \theta_i^t$
\end{algorithmic}
\end{algorithm}
```



step3的latex描述为：
```
\item \textbf{Abnormal Detection}

“Abnormal Detection”的主要过程基于局部离群因子（Local Outlier Factor, LOF）得到异常点。LOF算法是一种基于密度的异常检测方法，算法的核心思想是通过计算一个数值$lof_i$来反应节点$i$的异常程度。$lof_i$大致意思是节点$i$目的意图点$O_i$周围意图点所处位置的平均密度比上$O_i$所在位置的密度。比值越大于$1$，则$O_i$所在位置的密度越小于周围其他意图点所在位置的密度，即$O_i$越有可能是异常点。首先定义意图点$O_i$的$\kappa$距离$\widetilde{dis_i^\kappa}$，假设高维空间中存在节点$j$的意图点$O_j$，则有$O_j$与$O_i$之间的距离为$\|O_i-O_j\|$，如果满足以下两个条件，我们就认为$\widetilde{dis_i^\kappa}=\|O_i-O_j\|$：

\begin{itemize}
    \item 在样本空间中，至少存在$\kappa$个意图点$O_j'$，使得$\|O_i-O_j'\|\leq \|O_i-O_j\|$, 其中$j'\neq i$；
    \item 在样本空间中，至多存在$\kappa-1$个意图点$O_j'$，使得$\|O_i-O_j'\|< \|O_i-O_j\|$, 其中$j'\neq i$。
\end{itemize}

总的来说，$O_i$的$\kappa$距离$\widetilde{dis_i^\kappa}$表示高维空间中距离第$\kappa$远的点。之后定义意图点$O_i$的第$\kappa$距离邻域$Nei_i$为到$O_i$距离不超过$\widetilde{dis_i^\kappa}$的所有意图点的集合。由于可能同时存在多个第$\kappa$距离的数据，所以$|Nei_i|\geq \kappa$。可以想象，离群度越大的意图点的$\kappa$距离往往较大，离群度越小的意图点的$\kappa$距离往往较小。之后定义意图点$O_i$相对于意图点$O_j$的可达距离：

\begin{equation}
rea_{i,j}=\max\{\widetilde{dis_i^\kappa}, \|O_i-O_j\|\},
\end{equation}

也就是说，如果意图点$O_j$远离意图点$O_i$，则两者之间的可达距离就是他们之间的实际距离$\|O_i-O_j\|$；而如果二者距离足够近，则可达距离用意图点$O_i$的$\kappa$距离$\widetilde{dis_i^\kappa}$代替。之后定义意图点$O_i$的局部可达密度$lrd_i$为其$Nei_i$所有意图点的平均可达距离的倒数，即：

\begin{equation}
lrd_i=1/(\frac{\sum_{j\in Nei_i} rea_{i,j}}{|Nei_i|})
\end{equation}

此时，若有重复点，则可能导致$lrd$变为无限大。$lrd_i$的可以理解为意图点$O_i$所处位置的密度，密度越高则意图点$O_i$越有可能属于同一簇，密度越低则意图点$O_i$越有可能是离群点。也就是说，如果意图点$O_i$和周围邻域点是同一簇，则可达距离可能为较小的$\widetilde{dis_i^\kappa}$，导致可达距离之和较小，密度值较高；如果$O_i$和周围邻居意图点较远，则可达距离可能会取较大的$\|O_i-O_j\|$，导致可达距离之和较大，密度值较低，越有可能是离群点。最后，我们定义意图点$O_i$的局部离群因子$lof_i$为其$Nei_i$所有意图点的局部可达密度与其自身局部可达密度的比值的平均值，即：

\begin{equation}
    lof_i=\frac{\sum_{j\in Nei_i}\frac{lrd_j}{lrd_i}}{|Nei_i|}=\frac{\sum_{j\in Nei_i} lrd_j}{|Nei_i|\cdot lrd_i}
\end{equation}

如果$lof_i$比较接近$1$，则说明意图点$O_i$与其邻域点密度差不多，$O_i$可能和邻域属于同一簇；如果$lof_i$小于$1$，则说明意图点$O_i$的密度高于其邻域点的密度，$O_i$为密集点；如果$lof_i$大于$1$，则说明意图点$O_i$的密度低于其邻域点的密度，$O_i$可能是离群点。总之，LOF算法主要通过比较每个意图点$O_i$和其邻域点的密度来判断$O_i$是否为离群点，密度越低，则越有可能是离群点。而密度主要是通过点之间的距离来计算的，点之间的距离越远密度越低，距离越近密度越高。

计算每个节点$i$的局部离群因子$lof_i$，将$lof_i$大于$1$的客户端视为异常客户端，并将其梯度丢弃。
```
重写并返回step3这一部分的伪代码




简略一点，有公式的话描述可以少一点




简写这段代码

```
\For {Client $i$}
    \State Compute $lof_i = \frac{\sum_{j \in Nei_i} \frac{lrd_j}{lrd_i}}{|Nei_i|}$
    \If {$lof_i > 1$}
        \State Mark client $i$ as malicious, add to $U_{mal}$
    \Else
        \State Mark client $i$ as normal, add to $U_{nor}$
    \EndIf
\EndFor
```
不使用For、if-else，直接U_{mal}={xxx }





这部分写好的内容编译出来是：
```
verview: 前面的攻击中已经可以有效地针对单
轮次攻击进行识别，本部分主要结合客户端的历史记录
对客户端进行有效的识别。本部分将保留历史 T 轮次
的全局模型 θt′
g ，其中 max{t − T + 1, 1} ≤ t′ ≤ t，并将
其 Flatten 后视为高维空间中的一个点；同时保留每个
客户端的历史更新变化，将其视为高维空间中的一个向
量，将其延伸后视为一个射线。对于每个客户端，使用
最小覆盖球算法计算得到一个最小的超球，覆盖 ζ 比
例的射线。最终球心的位置可以被视为对应客户端对全
局模型的意图引导点，球心的半径可以视为对应客户端
的置信度。
算法 2展示了根据每个客户端历史记录进行识别的
识别过程，主要包含 4 个步骤：
1) Keep T-rounds History
“Keep T-rounds History”的主要算法过程如算法
2的 3-13 行所示。像正常的联邦学习过程一样，中
央服务器每次下发一个全局模型，每个客户端得到
全局模型后使用本地数据进行训练，并将更新变化
上传到中央服务器中。我们将第 t 轮次的全局模型
记为 θt，将客户端 i 第 t 轮次训练后的模型记为
θt
i 。客户端 i 将训练后的模型 θt
i 减去训练前中央
服务器下发的全局模型 θt，就得到了 t 轮次的更新
变化 ▽θt
i = θt
i − θt。客户端将更新变化 ▽θt
i 上传
到中央服务器，中央服务器在聚合的同时，记录下
每个客户端当前轮次的更新 ▽θt
i 展平后的结果  ̄θt
i ，
同时存下中央服务器上轮次下发的全局模型 θt−1。
中央服务器最多保留 T 轮次的模型和更新历史记
录。此部分可以参考算法 2的 step1。
2) Obtain the Purpose Intention
“Obtain the Purpose Intention”的主要算法过程
如算法 2的 14-18 行所示。这个问题可以抽象为高
维空间中的一些具有起点的射线。射线的起点代表
上轮次的全局模型，射线的方向代表本轮次展平后
的更新变化。问题的优化目标是：找到一个最小的
超球，至少覆盖 ζ 比例的射线。此部分又可以分为
“Construct Ray Model”和“Minimum Enclosing
Hypersphere Calculation”两部分。
“Construct Ray Model”的主要过程如下。对于
每 个 客 户 端 Ri， 中 央 服 务 器 保 留 了 最 近 T 轮
的全局模型历史以及每轮次的更新变化历史。设
θt−T
g , θt−T +1
g , . . . , θt−1
g 是最近 T 轮次的全局模型，
且这些全局模被“展平”到了高维空间中的一个
点。类似地，设 ▽θt−T
i , ▽θt−T +1
i , . . . , ▽θt−1
i 是最近
T 轮次客户端 i 上传的展平后的梯度变化向量。对
于每一轮 t′ (max{t − T + 1, 1} ≤ t′ ≤ t)，我们将
每个射线的起点表示为展平后的全局模型 θt′
g ，并
将每个射线的方向向量表示为展平后的梯度变化
▽θt′
i 。因此，射线模型可以构建如下：
 ̃θt′ −1
i = F latten(θt′ −1
g ), (2)
vt′
i = ▽θt′
i , (3)
其中， ̃θt′ −1
i 是射线的起点，vt′
i 是射线的方向向量。
则射线方程为：
lt′
i =  ̃θt′ −1
i + αvt′
i , (4)
其中，α ∈ [0, +∞) 为射线的自变量参数。
“Minimum Enclosing Hypersphere Calculation”的
主要过程如下。基于构建的射线模型，使用最小覆
盖球算法找到一个能够覆盖至少 ζ 比例的射线的
最小超球。首先，设定初始球心 Oi,0 为所有射线起
点的几何中心，计算公式如下：
Oi,0 = 1
T
t∑
t′ =max{t−T +1,1}
 ̃θt′ −1
g . (5)
初始半径 ri,0 设定为从初始球心 Oi,0 到所有射线
起点的最大距离：
ri,0 = t
max
t′ =max{t−T +1,1}
∥Oi,0 −  ̃θt′ −1
g ∥. (6)
接下来，采用迭代方法来更新球心和半径：
对于每次迭代 k，计算当前球心 Oi,k 到所有射线
的最近点  ̃Ot′
i,k。对于每个节点 Ri 的第 t′(max{t −
T + 1, 1} ≤ t′ ≤ t) 轮，对于射线 lt′
i ，找到球心 Oi,k
与当前射线的投影点 ˆOt′
i,k，并计算参数 αk：
αk = (Oi,k −  ̃θt′ −1
i ) · vt′
i
vt′
i · vt′
i
, (7)
则最近点  ̃Ot′
i,k =  ̃θt′ −1
i + max {0, αk} vt′
i 。
计算当前球心 Oi,k 到这些最近点  ̃Ot′
i,k 的距离集合
Disi,k = {∥Oi,k −  ̃Ot′
i,k∥, max{t−T +1, 1} ≤ t′ ≤ t}，
并按从小到大的顺序进行排序，然后暂时舍弃 1 − ζ
比例的射线来忽略离群点对计算过程的影响，得到
集合 Dis′
i,k，保留射线对应轮次的集合记为  ̃T 。将
球心向着被保留的最近点中距离最远的点  ̃Ot′
max
i,k
移动：
Oi,k+1 = Oi,k + η′(  ̃Ot′
max
i,k − Oi,k), (8)
其中 η′ 为学习率。移动后的球心半径为：
ri,k+1 = {max{∥Oi,k+1 −  ̃Ot′
i,k+1∥}, t′ ∈  ̃T }, (9)
其中  ̃Ot′
i,k+1 为更新后的球心 Oi,k+1 到被保留射线
的最近点。终止条件：当更新后的半径 ri,k+1 与上
一次的半径 ri,k 相差小于一个预设的阈值 λ，或者
达到预设的最大迭代次数 kmax 时，停止迭代。这
部分可以定义为一个有约束的最优化问题：
min
Oi ,ri
ri (10)
s.t.
∣
∣
∣
{
t′ | ∥Oi −  ̃Ot′
i ∥ ≤ ri
}∣
∣
∣ ≥ ζT. (11)
3) Abnormal Detection
“Abnormal Detection”的主要过程基于局部离群因
子（Local Outlier Factor, LOF）得到异常点。LOF
算法是一种基于密度的异常检测方法，算法的核心
思想是通过计算一个数值 lofi 来反应节点 i 的异
常程度。lofi 大致意思是节点 i 目的意图点 Oi 周
围意图点所处位置的平均密度比上 Oi 所在位置的
密度。比值越大于 1，则 Oi 所在位置的密度越小
于周围其他意图点所在位置的密度，即 Oi 越有可
能是异常点。首先定义意图点 Oi 的 κ 距离  ̃disκ
i ，
假设高维空间中存在节点 j 的意图点 Oj ，则有 Oj
与 Oi 之间的距离为 ∥Oi − Oj ∥，如果满足以下两
个条件，我们就认为  ̃disκ
i = ∥Oi − Oj ∥：
• 在样本空间中，至少存在 κ 个意图点 O′
j ，使得
∥Oi − O′
j ∥ ≤ ∥Oi − Oj ∥, 其中 j′ ̸ = i；
• 在样本空间中，至多存在 κ − 1 个意图点 O′
j ，
使得 ∥Oi − O′
j ∥ < ∥Oi − Oj ∥, 其中 j′ ̸ = i。
总的来说，Oi 的 κ 距离  ̃disκ
i 表示高维空间中距离
第 κ 远的点。之后定义意图点 Oi 的第 κ 距离邻
域 N eii 为到 Oi 距离不超过  ̃disκ
i 的所有意图点的
集合。由于可能同时存在多个第 κ 距离的数据，所
以 |N eii| ≥ κ。可以想象，离群度越大的意图点的
κ 距离往往较大，离群度越小的意图点的 κ 距离往
往较小。之后定义意图点 Oi 相对于意图点 Oj 的
可达距离：
reai,j = max{  ̃disκ
i , ∥Oi − Oj ∥}, (12)
也就是说，如果意图点 Oj 远离意图点 Oi，则两者之
间的可达距离就是他们之间的实际距离 ∥Oi − Oj ∥；
而如果二者距离足够近，则可达距离用意图点 Oi
的 κ 距离  ̃disκ
i 代替。之后定义意图点 Oi 的局部
可达密度 lrdi 为其 N eii 所有意图点的平均可达距
离的倒数，即：
lrdi = 1/(
∑
j∈N eii reai,j
|N eii| ) (13)
此时，若有重复点，则可能导致 lrd 变为无限大。
lrdi 的可以理解为意图点 Oi 所处位置的密度，密
度越高则意图点 Oi 越有可能属于同一簇，密度越
低则意图点 Oi 越有可能是离群点。也就是说，如
果意图点 Oi 和周围邻域点是同一簇，则可达距离
可能为较小的  ̃disκ
i ，导致可达距离之和较小，密度
值较高；如果 Oi 和周围邻居意图点较远，则可达
距离可能会取较大的 ∥Oi − Oj ∥，导致可达距离之
和较大，密度值较低，越有可能是离群点。最后，我
图 2. Benign and Malicious Nodes Before Aggregation
们定义意图点 Oi 的局部离群因子 lofi 为其 N eii
所有意图点的局部可达密度与其自身局部可达密度
的比值的平均值，即：
lofi =
∑
j∈N eii
lrdj
lrdi
|N eii| =
∑
j∈N eii lrdj
|N eii| · lrdi
(14)
如果 lofi 比较接近 1，则说明意图点 Oi 与其邻域
点密度差不多，Oi 可能和邻域属于同一簇；如果
lofi 小于 1，则说明意图点 Oi 的密度高于其邻域
点的密度，Oi 为密集点；如果 lofi 大于 1，则说
明意图点 Oi 的密度低于其邻域点的密度，Oi 可能
是离群点。总之，LOF 算法主要通过比较每个意图
点 Oi 和其邻域点的密度来判断 Oi 是否为离群点，
密度越低，则越有可能是离群点。而密度主要是通
过点之间的距离来计算的，点之间的距离越远密度
越低，距离越近密度越高。
计算每个节点 i 的局部离群因子 lofi，将 lofi 大于
1 的客户端视为异常客户端，并将其梯度丢弃。
4) Gradients Aggregation
剔除了异常节点后，依据正常节点的置信度 cre 进
行聚合。其中定义：
crei = 1
ri + ρ , (15)
ρ 是一个很小的正数，以防分母为 0。
如图 2所示，使用 LOF 算法识别恶意客户端可以
剔除潜在的恶意节点，但是在正常的节点中，同样
存在一些置信度很低的节点。这些节点的最小覆盖
球甚至可能和异常节点有一定交集。因此，在聚合
的过程中，这些置信度较低的点所占的权重就应该
越低。
试想，假设有两个节点的最小覆盖超球的半径都很
小，但是相差倍数很高，那么它们计算出来的置信
8
度相差倍数也会很高。但其实它们的意图点都十分
明确，因此权重应该都比较高且相差不应很大。所
以我们可以使用激活函数 Tanh 来对置信度进行加
权处理：
cre′
i = tanh(crei), (16)
激活函数 Tanh 在 0 到 ∞ 范围内是一个上升又快
到慢的单调递增函数。当 crei → 0 时 cer′
i → 0，也
就是说节点 i 的最小覆盖球半径 ri 很大时该节点
在聚合时的权重很小；当 crei → ∞ 时 cer′
i → 1，
也就是说节点 i 的最小覆盖球半径 ri 很小时该节
点在聚合时的权重较大。这样，我们就可以对每个
加权处理过的置信度 cer′
i 进行归一化处理，得到
权重 wi。其中：
wi = cre′
i
∑n
i=1 cre′
i
, (17)
因此第 t 轮模型的梯度聚合公式为：
θt
g = θt−1
g +
n∑
i=1
```

假设我已经有了第一步的数据，请给出代码实现第二步、第三步和第四步。






将这些点绘图，并标注异常点





1. 保存为svg
2. 直接生成一些模拟数据






生成的数据里，设置大约15%的异常用户




git只clone一个分支，不要fetch全部




clone之后，将其作为另一个仓库的另一个分支并推送




我在windowns主机上clone好了这个仓库，我有一台只能访问局域网的Linux服务器，上面的对应文件在`lzy@3090.narc.letmefly.xyz:/home/lzy/ltf/Codes/LLM/wb2`文件夹下。
我想写一个脚本Let.bat，当运行这个命令时，依据.gitignore中的内容强制更新服务器上的内容。





不，它的意思是说bash里没有rsync命令




这个报错是我在windows上执行`bash -c "rsync `命令导致的，windows上的git bash并不支持`rsync`




Cygwin  安装  rsync




使用LORA微调CLIP怎么设置学习率



为高一学生写一篇英语演讲比赛演讲稿，主题是the power of Dreams，要求三分钟以内。



我要ubuntu安装新的cuda驱动，需要先卸载旧驱动。卸载旧驱动需要先关掉占用进程，这些进程被关掉后会立刻自动重启。



一般微调多少轮



<!-- 目前看来比较有挑战的是关键词提取还有从PDF里提取作者摘要这些信息 -->
<!-- 关键词提取应该有现成的库，不知道效果咋样 -->
然后从PDF里面提作者信息的话有点难度，一个是要使用python读取pdf的内容，一个是







关于假阳率、真阴率等的衡量，有F1-Score、误封率等衡量指标。除了这两个指标还有哪些指标？




我准备使用：“准确率”、精确率 召回率  特异度  误报率  假阴率  
将其总结成一个表格，表格内容包括计算公式、说明。说明要包含越大越好还是越小越好。





先介绍TP、TN、FP、FN都是什么，然后将“越大越好还是越小越好”融合到“说明”里，返回latex源码




我使用的是两栏式布局，这个表格超出了单侧跑到另外一侧去了。





介绍这部分的含义
```
\begin{table}[h!]
\small
\centering
\begin{tabular}{|m{3cm}|m{4cm}|m{6.5cm}|}
```





能否不指定列宽






这样表格直接地特别宽，直接超过两栏了。能否指定表格的总宽度为页面总宽度的一半？




latex画表格，






python画热力图






conda设置超时时间



conda env create --name clip_lora --file clip_lora.yaml
一直卡在Collecting package metadata (repodata.json): 





能不能手动下载repodata.json并保存到本地





windows开启ssh远程连接




windows ssh开启免密登录





为什么要这样设置权限
```
设置文件权限： 确保 authorized_keys 文件的权限正确。你可以使用 PowerShell 或文件资源管理器来设置权限，使得只有文件所有者可以读取或写入：

powershell
复制代码
icacls C:\Users\<YourUsername>\.ssh\authorized_keys /inheritance:r
icacls C:\Users\<YourUsername>\.ssh\authorized_keys /grant <YourUsername>:(R)
```






我的用户名是admin，我应该怎么写




PS C:\Users\admin\.ssh> icacls C:\Users\admin\.ssh\authorized_keys /inheritance:r
已处理的文件: C:\Users\admin\.ssh\authorized_keys
已成功处理 1 个文件; 处理 0 个文件时失败
PS C:\Users\admin\.ssh> icacls C:\Users\admin\.ssh\authorized_keys /grant admin:(R)
icacls C:\Users\admin\.ssh\authorized_keys /inheritance:r
无效参数“admin:”






<!-- icacls C:\Users\admin\.ssh\authorized_keys /grant "desktop-5oj3khu\admin":(R) -->





PS C:\Users\admin\.ssh> icacls C:\Users\admin\.ssh\authorized_keys /grant "desktop-5oj3khu\admin":(R)
whoami
无效参数“desktop-5oj3khu\admin”





VsCode远程服务器python不高亮





conda能不能将一个环境中的一个包复制到另一个环境中




```
(pytorch) C:\Users\admin\Desktop\LLM\wb2\Codes>pip list | findstr torch
torch                1.8.1+cu111
torchaudio           0.8.1
torchvision          0.9.1+cu111

(pytorch) C:\Users\admin\Desktop\LLM\wb2\Codes>conda activate clip_lora

(clip_lora) C:\Users\admin\Desktop\LLM\wb2\Codes>pip list | findstr torch 
torch              2.4.1
torchvision        0.19.1
```

我想把`clip_lora`环境中的torch相关的包变成`pytorch`环境中的版本。





```
(clip_lora) C:\Users\admin\Desktop\LLM\wb2\Codes>pip uninstall torch torchvision torchaudio
Found existing installation: torch 2.4.1
Uninstalling torch-2.4.1:
  Would remove:
    c:\programdata\anaconda3\envs\clip_lora\lib\site-packages\functorch\*
    c:\programdata\anaconda3\envs\clip_lora\lib\site-packages\torch-2.4.1.dist-info\*
    c:\programdata\anaconda3\envs\clip_lora\lib\site-packages\torch\*
    c:\programdata\anaconda3\envs\clip_lora\lib\site-packages\torchgen\*
    c:\programdata\anaconda3\envs\clip_lora\scripts\convert-caffe2-to-onnx.exe
    c:\programdata\anaconda3\envs\clip_lora\scripts\convert-onnx-to-caffe2.exe
    c:\programdata\anaconda3\envs\clip_lora\scripts\torchrun.exe
Proceed (Y/n)?
  Successfully uninstalled torch-2.4.1
Found existing installation: torchvision 0.19.1
Uninstalling torchvision-0.19.1:
  Would remove:
    c:\programdata\anaconda3\envs\clip_lora\lib\site-packages\torchvision-0.19.1.dist-info\*
    c:\programdata\anaconda3\envs\clip_lora\lib\site-packages\torchvision\*
Proceed (Y/n)? y
  Successfully uninstalled torchvision-0.19.1
WARNING: Skipping torchaudio as it is not installed.

(clip_lora) C:\Users\admin\Desktop\LLM\wb2\Codes>pip list | findstr torch

(clip_lora) C:\Users\admin\Desktop\LLM\wb2\Codes>pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
Looking in indexes: https://mirrors.ustc.edu.cn/pypi/simple
Looking in links: https://download.pytorch.org/whl/torch_stable.html
ERROR: Could not find a version that satisfies the requirement torch==1.8.1+cu111 (from versions: 1.11.0, 1.11.0+cpu, 1.11.0+cu113, 1.11.0+cu115, 1.12.0, 1.12.0+cpu, 1.12.0+cu113, 
1.12.0+cu116, 1.12.1, 1.12.1+cpu, 1.12.1+cu113, 1.12.1+cu116, 1.13.0, 1.13.0+cpu, 1.13.0+cu116, 1.13.0+cu117, 1.13.1, 1.13.1+cpu, 1.13.1+cu116, 1.13.1+cu117, 2.0.0, 2.0.0+cpu, 2.0.0+cu117, 2.0.0+cu118, 2.0.1, 2.0.1+cpu, 2.0.1+cu117, 2.0.1+cu118, 2.1.0, 2.1.0+cpu, 2.1.0+cu118, 2.1.0+cu121, 2.1.1, 2.1.1+cpu, 2.1.1+cu118, 2.1.1+cu121, 2.1.2, 2.1.2+cpu, 2.1.2+cu118, 2.1.2+cu121, 2.2.0, 2.2.0+cpu, 2.2.0+cu118, 2.2.0+cu121, 2.2.1, 2.2.1+cpu, 2.2.1+cu118, 2.2.1+cu121, 2.2.2, 2.2.2+cpu, 2.2.2+cu118, 2.2.2+cu121, 2.3.0, 2.3.0+cpu, 2.3.0+cu118, 2.3.0+cu121, 2.3.1, 2.3.1+cpu, 2.3.1+cu118, 2.3.1+cu121, 2.4.0, 2.4.1)
ERROR: No matching distribution found for torch==1.8.1+cu111
```





我在`pytorch`环境中已经安装好了对应的版本，能不能直接使用这个环境中的内容，而不是再从网上下载安装





```
(clip_lora) C:\Users\admin\Desktop\LLM\wb2\Codes>python
Python 3.10.14 | packaged by Anaconda, Inc. | (main, May  6 2024, 19:44:50) [MSC v.1916 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "C:\ProgramData\Anaconda3\envs\clip_lora\lib\site-packages\torch\__init__.py", line 213, in <module>
    raise ImportError(textwrap.dedent('''
ImportError: Failed to load PyTorch C extensions:
    It appears that PyTorch has loaded the `torch/_C` folder
    of the PyTorch repository rather than the C extensions which
    are expected in the `torch._C` namespace. This can occur when
    using the `install` workflow. e.g.
        $ python setup.py install && python -c "import torch"

    This error can generally be solved using the `develop` workflow
        $ python setup.py develop && python -c "import torch"  # This should succeed
    or by running Python from a different directory.
```




把这个表转置一下
```
\[
\begin{array}{|c|c|c|}
\hline
\text{参数} & \text{MNIST} & \text{CIFAR-10} \\
\hline
\text{学习率} & 0.01 & 0.01 \\
\text{全局轮次} & 50 & 50 \\
\text{本地批次} & 5 & 5 \\
\text{客户端数量} & 20 & 20 \\
\text{批次大小} & 64 & 64 \\
\text{优化器} & \text{SGD} & \text{SGD} \\
\text{模型} & \text{ViT} & \text{ViT} \\
\hline
\end{array}
\]
```





返回对应的latex源码





这段代码有什么错
```
\begin{table}[h!]
\small
\centering
\begin{tabularx}{0.5\textwidth}{|X|X|X|X|X|X|X|X|}
\hline
& \text{学习率} & \text{全局轮次} & \text{本地批次} & \text{客户端数量} & \text{批次大小} & \text{优化器} & \text{模型} \\
\hline
\text{MNIST} & 0.01 & 50 & 5 & 20 & 64 & \text{SGD} & \text{ViT} \\
\text{CIFAR-10} & 0.01 & 50 & 5 & 20 & 64 & \text{SGD} & \text{ViT} \\
\hline\end{tabularx}
\caption{参数取值表}
\end{table}
```




我使用的是两栏式latex，如何让这个表格横跨两栏





这段代码有什么错
```
\begin{table*}[h!]
\small
\centering
\begin{tabularx}{textwidth}{|X|X|X|X|X|X|X|X|}
\hline
& 学习率 & 全局轮次 & 本地批次 & 客户端数量 & 批次大小 & 优化器 & 模型 \\
\hline
MNIST & 0.01 & 50 & 5 & 20 & 64 & SGD & ViT \\
CIFAR-10 & 0.01 & 50 & 5 & 20 & 64 & SGD & ViT \\
\hline
\end{tabularx}
\caption{参数取值表}
\end{table*}
```





如何引用网页




引用这个`https://huggingface.co/openai/clip-vit-base-patch32`





peft的get_peft_model(base_model, config)是复制了一份还是指向了同一内存





```
base_model = CLIPModel.from_pretrained('../Datasets/clip-vit-base-patch32')
self.local_model = get_peft_model(base_model, config)
self.local_model.print_trainable_parameters()
base_model = CLIPModel.from_pretrained('../Datasets/clip-vit-base-patch32')
self.target_model = get_peft_model(base_model, config)
self.target_model.print_trainable_parameters()
```
这样写有问题吗




介绍MR攻击




联邦学习中使用余弦相似度进行恶意检测的论文叫什么





解释这篇文章《ON THE BYZANTINE ROBUSTNESS OF CLUSTERED FEDERATED LEARNING》
```
Federated Learning (FL) is currently the most widely adopted
framework for collaborative training of (deep) machine learning models under privacy constraints. Albeit it’s popularity,
it has been observed that Federated Learning yields suboptimal results if the local clients’ data distributions diverge. The
recently proposed Clustered Federated Learning Framework
addresses this issue, by separating the client population into
different groups based on the pairwise cosine similarities between their parameter updates. In this work we investigate
the application of CFL to byzantine settings, where a subset
of clients behaves unpredictably or tries to disturb the joint
training effort in an directed or undirected way. We perform
experiments with deep neural networks on common Federated Learning datasets which demonstrate that CFL (without
modifications) is able to reliably detect byzantine clients and
remove them from training
```





我想要使用python的matplotlib画图，一张图上一共两行四列8个子图。
第一行是攻击A，第二行是攻击B。
第一列是余弦相似度检测，第二列是fltrust，第三列是flame，第四列是SecFFT。
每个子图都是一张热力图，用来展示每种检测方法对于恶意用户的识别区分准结果。
图中使用全英文描述即可。
目前一共有20个客户端，前4个客户端是恶意客户端，后面是良性客户端。
请模拟一些数据，并返回python源码，不需要plt.show，直接保存为jpg即可。




解释`np.random.uniform`






我只有每个客户端的评分，并没有两两之间的相似度评分。





我虽然只有单个客户端的评分，但是我还想画成两两之间对比的图。





关于这个表格，请添加FP、TP、FN、TN、ROC、AUC、MCC，并将表格转置，变成横跨两栏的表格。




表格不要横向显示，正常纵向显示即可




介绍ROC曲线





现在我要画一个latex表格，有一些列，包括：FP、TP、FN、TN、准确率、精确率、召回率、特异度、误报率、假阴率、AUC、MCC

还有一些行，分成两大行，第一行是攻击方式1，第二行是攻击方式2；对于每种攻击方式，分为三个小行，分别防御方式1、防御方式2、防御方式3。

先返回渲染好后的结果，暂时不需要返回源码。





攻击方式写成类似这样的：

-----+--------
     | 防御1
攻击1| 防御2
     | 防御3
-----+--------
     | 防御1
攻击2| 防御2
     | 防御3
-----+--------






是的，返回其latex源码。不需要begin{document}，只需要表格相关源码即可。




<recently read> \endtemplate 
                             
l.911 \end{tabularx}
                    
You have given more \span or & marks than there were
in the preamble to the \halign or \valign now in progress.
So I'll assume that you meant to type \cr instead.




See the LaTeX manual or LaTeX Companion for explanation.
Type  H <return>  for immediate help.
 ...                                              
                                                  
l.907 \end{tabularx}
                    
You've lost some text.  Try typing  <return>  to proceed.
If that doesn't work, type  X <return>  to quit.





See the LaTeX manual or LaTeX Companion for explanation.
Type  H <return>  for immediate help.
 ...                                              
                                                  
l.907 \end{tabularx}
                    
You've lost some text.  Try typing  <return>  to proceed.
If that doesn't work, type  X <return>  to quit.





使用latex画一个表格：

攻击方式 |防御方式 | 值
-----+--------
     | 防御1
攻击1| 防御2
     | 防御3
-----+--------
     | 防御1
攻击2| 防御2
     | 防御3
-----+--------

返回其latex源码






这段代码是可行的：
```
\begin{tabular}{|c|c|c|}
\hline
攻击方式 & 防御方式 & 值 \\
\hline
\multirow{3}{*}{攻击1} & 防御1 &  \\ 
                       & 防御2 &  \\ 
                       & 防御3 &  \\ 
\hline
\multirow{3}{*}{攻击2} & 防御1 &  \\ 
                       & 防御2 &  \\ 
                       & 防御3 &  \\ 
\hline
\end{tabular}
```
参考这段代码修改BUG







写一段python代码，假设我一共有20个客户端，前4个客户端是恶意客户端。我识别出来的恶意客户端的结果是一个数组。请你据此计算出表格中的每一列。






```
def flame(trained_params, current_model_param, param_updates):
    # === clustering ===
    trained_params = torch.stack(trained_params).double()
    cluster = hdbscan.HDBSCAN(
        metric="cosine",
        algorithm="generic",
        min_cluster_size=args.participant_sample_size // 2 + 1,
        min_samples=1,
        allow_single_cluster=True,
    )
    cluster.fit(trained_params)
    predict_good = []
    for i, j in enumerate(cluster.labels_):
        if j == 0:
            predict_good.append(i)
    k = len(predict_good)

    # === median clipping ===
    model_updates = trained_params[predict_good] - current_model_param
    local_norms = torch.norm(model_updates, dim=1)
    S_t = torch.median(local_norms)
    scale = S_t / local_norms
    scale = torch.where(scale > 1, torch.ones_like(scale), scale)
    model_updates = model_updates * scale.view(-1, 1)

    # === aggregating ===
    trained_params = current_model_param + model_updates
    trained_params = trained_params.sum(dim=0) / k

    # === noising ===
    delta = 1 / (args.participant_sample_size**2)
    epsilon = 10000
    lambda_ = 1 / epsilon * (math.sqrt(2 * math.log((1.25 / delta))))
    sigma = lambda_ * S_t.numpy()
    print(
        f"sigma: {sigma}; #clean models / clean models: {k} / {predict_good}, median norm: {S_t},"
    )
    trained_params.add_(torch.normal(0, sigma, size=trained_params.size()))

    # === bn ===
    global_update = dict()
    for i, (name, param) in enumerate(param_updates.items()):
        if "num_batches_tracked" in name:
            global_update[name] = (
                1 / k * param_updates[name][predict_good].sum(dim=0, keepdim=True)
            )
        elif "running_mean" in name or "running_var" in name:
            local_norms = torch.norm(param_updates[name][predict_good], dim=1)
            S_t = torch.median(local_norms)
            scale = S_t / local_norms
            scale = torch.where(scale > 1, torch.ones_like(scale), scale)
            global_update[name] = param_updates[name][predict_good] * scale.view(-1, 1)
            global_update[name] = 1 / k * global_update[name].sum(dim=0, keepdim=True)

    return trained_params.float().to(args.device), global_update
```

解释这段代码，flame防御






热力图中，原始数据怎样会显得热力图对比度更加明显，怎样会显得不明显







将这段话翻译成英文，并美化之
```
\textbf{Motivation}: 为了规避异常检测，有很多通过多轮次的组合攻击来植入后门的攻击。这些攻击很难在单轮次中检测出来，大致分为三类：1)限制大小的攻击，如\cite{geisler2024attacking, chen2024optimal}，这些攻击方法会限制单次攻击的大小，使得攻击并不是很明显，从而变得难以检测；2)限制角度的攻击，如\cite{you2023three, sagliano2024powered, dong2023adaptive}，这些攻击会限制每次攻击的角度，使其与正常的攻击类似从而变得难以识别；3)限制符号的攻击，如\cite{wan2023average, zhu2023boosting}，这些攻击通过梯度缩放或者梯度合成，控制恶意攻击的梯度与正常客户端梯度的正负维持一致，从而增加识别难度。本部分结合每个客户端的历史记录，通过识别攻击者最终的模型引导意图点来识别恶意节点。
```




还要写成一段话，返回英文及其latex源码




为了让我熬夜不困，别人送我了咖啡（表情包）。为了让她睡觉，应该送她什么？






将这段话翻译成英文，并美化之
```
% 我们将算法总结成了伪代码，如算法\ref{alg:malicious-node-detection-history}所示。其中第3-7行是步骤1，表示了保留T轮次历史记录的方法；第8-15行是步骤2，表示了使用最小覆盖超球算法求每个机器人节点意图点和置信度的方法；第16-27行是步骤3，表明了如何通过LOF算法求出正常节点和异常节点；第28-32行是步骤4，展示了依据每个机器人节点置信度进行加权聚合的方法。
```





基于置信度的安全聚合方式，用英语怎么说
其中置信度单词为certification的一个形式