<!-- 我有一个数组为torch.tensor类型的update，我想要创建一个一模一样大小的数组 -->




<!-- 我有10个客户端的梯度，我应该如何聚类 -->



我的update数组为torch.tensor类型，shape为torch.Size([10, 2674688])，代表10个客户端的梯度。
我赢如何使用余弦相似度？





能否不归一化，直接使用`smp.cosine_similarity`计算




python判断一个元素是否为torch.tensor




KMEANS根据每个客户端更新的梯度进行聚类




```
def kmeans_clustering(updates_np, n_clusters=2):
    # TODO: 补全这个函数

# 通过历史记录的梯度计算恶意客户端
def calc_maliciousAndCos_justByGrads(roundsNum: List[int], dirPath: str):
    clientPerRound = 10  # 这里就先写死了
    maliciousPerRound = 3
    gradients = [0] * len(roundsNum) * clientPerRound
    maliciouses = []
    for th, roundNum in enumerate(roundsNum):
        participants_thisRound = getParticipants(roundNum, dirPath)
        model_updates = loadPkl(roundNum, dirPath)
        keys = list(model_updates.keys())
        last_layer_updates = model_updates[keys[-2]]
        K = len(last_layer_updates)
        for i in range(K):
            thisParticipant = participants_thisRound[i]
            if thisParticipant < maliciousPerRound:
                thisIndex = th * maliciousPerRound + thisParticipant
            else:
                thisIndex = maliciousPerRound * len(roundsNum) + (clientPerRound - maliciousPerRound) * th + thisParticipant - maliciousPerRound
            gradients[thisIndex] = last_layer_updates[i].cpu().numpy()
        _, foolsgoldMaliciousIndex, _ = kmeans_clustering(gradients)
        for thisMaliciousIndex in foolsgoldMaliciousIndex:
            if thisMaliciousIndex < maliciousPerRound:
                thisIndex = th * maliciousPerRound + thisMaliciousIndex
            else:
                thisIndex = maliciousPerRound * len(roundsNum) + (clientPerRound - maliciousPerRound) * th + thisMaliciousIndex - maliciousPerRound
            maliciouses.append(thisIndex)
    shuffleArray = np.arange(50)
    np.random.shuffle(shuffleArray[0:len(roundsNum) * maliciousPerRound])
    np.random.shuffle(shuffleArray[len(roundsNum) * maliciousPerRound:50])
    malicious_shuffled = [shuffleArray[i] for i in maliciouses]
    gradients_shuffled = [gradients[shuffleArray[i]] for i in range(50)]
    cs = smp.cosine_similarity(gradients_shuffled)
    cs_tensor = torch.from_numpy(cs)
    return malicious_shuffled, cs_tensor
```




```
ong-term Attack Intention Detection
Motivation: To evade anomaly detection systems, many
backdoor attacks employ multi-round composite strategies that
are difficult to detect in a single round. These attacks can be
broadly classified into three types: (1) Size-Limited Attacks
(e.g., [28], [29]), which constrain the magnitude of each attack
to make it subtle and less conspicuous, thereby reducing its
detectability; (2) Angle-Limited Attacks (e.g., [30]–[32]),
which restrict the direction of each attack so that it appears
similar to legitimate updates, making them harder to recognize;
and (3) Sign-Limited Attacks (e.g., [33], [34]), which use
techniques such as gradient scaling or gradient composition
to ensure that the signs of the malicious gradients align with
those of legitimate clients, thereby increasing the difficulty
of detection. This section presents an approach that leverages
Algorithm 1 Instantaneous Attack Behavior Perception
1: Input: d, N , (▽θt
0, ▽θt
1, . . . , ▽θt
N ) ∈ RN ×d, m ▷
d is the dimension of each node update; N is the
number of the nodes participating during each round;
(▽θt
0, ▽θt
1, . . . , ▽θt
N ) ∈ RN ×d is the local updates from
nodes during the t-th round; m is the length of low-
frequency components
2: Output: Unor , Umal ▷ benign nodes, malicious nodes
3: for Ri ∈ (R1, . . . , RN ) do
4:  ̄θt
i ← F latten(▽θt
i )
5: Gt
i ← T runc(DCT ( ▽θt
i
∥▽θt
i ∥2
)), m) ▷ Compute DCT
and retain only the top m low-frequency components
6: end for
7: (Ct
0, Ct
1, . . . , Ct
κ) ← Clustering(▽θt
0, ▽θt
1, . . . , ▽θt
N ) ▷
κ denotes the number of clusters
8: Cmax ← arg max
Ci
|Ci|, i = 1, 2, . . . , κ ▷ |Ci| denotes
the number of nodes in cluster Ci
9: Ht ← (Gt
0, Gt
1, . . . , Gt
n) ∈ Rm×n ▷ Stacking to form
Matrix Ht, where Ri0 , Ri1 , . . . , Rin ∈ Cmax.
10: ˆHt ← (Ht)T Ht
11: λt
max, ξt
max ← eig( ˆHt) ▷ Calculating the maximum
singular value and its corresponding eigenvector
12:  ̃Gt ← Htξt
max√λt
max
▷ The clean ingredient
13: for Ri ∈ (R1, . . . , RN ) do
14: Chi2
i ← 1
2
∑m−1
k=0
(Gt
i [k]−  ̃Gt[k])2
|Gt
i [k]|+|  ̃Gt[k]|+ε
15: end for
16: S ← {Chi1, Chi2, . . . , Chin} ▷ The Distance
differences calculated by Chi-square distance.
17: {C1, C2} ← KM eans(S, 2) ▷ Cluster S into 2 clusters
using KMeans.
18: Unor ← Cmax, Umal ← {Ci | i ̸ = max}
each client’s historical records to identify malicious nodes by
analyzing their ultimate intent to manipulate the global model.
Overview: The previous defense mechanisms have effec-
tively identified single-round attacks; this section focuses on
leveraging clients’ historical records for effective identifica-
tion. We maintain the global models θt′
g from the past T
rounds, where max{t − T + 1, 1} ≤ t′ ≤ t, and consider their
flattened versions as points in a high-dimensional space. Si-
multaneously, we keep track of each client’s historical updates,
viewing them as vectors in the high-dimensional space and
extending them as rays. For each client, we use the Minimum
Enclosing Ball algorithm to compute a minimal hypersphere
that covers a ζ proportion of the ray. The final position of the
hypersphere’s center can be regarded as the intended steering
point of the global model for the corresponding client, and the
radius of the hypersphere can be considered the confidence
level of that client.
This section mainly consists of four steps:
1) Construction of Weights updates and model databases :
As in a typical federated learning process, the central server
distributes a global model in each round. Upon receiving the
global model, each client trains on its local data and uploads
the updates to the central server. Let the global model in the
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 6
Algorithm 2 Malicious Node Detection and Gradient Aggre-
gation
1: Input: Global model history {θt−T
g , . . . , θt−1
g }, historical
gradient updates for each client {▽θt−T
i , . . . , ▽θt−1
i }
2: Output: Sets of normal clients Unor and malicious clients
Umal
3: Step 1: Keep T-rounds History
4: for Client i do
5:  ̄θt
i ← F latten(▽θt
i ) ▷ Record flattened gradients and
models θt−1
g
6:  ̃θt′ −1
i ← F latten(θt′ −1
g ), vt′
i ← ▽θt′
i , max{t − T +
1, 1} ≤ t′ ≤ t ▷ Construct rays
7: end for
8: Step 2: Obtain the Purpose Intention
9: for Client i do
10: Initialize center Oi,0 and radius ri,0
11: while Not Converged do
12: Update center Oi,k+1 and radius ri,k+1
13: end while
14: crei ← 1
ri+ρ ▷ Calculate confidence
15: end for
16: Step 3: Abnormal Detection
17: for Client i do
18: for Each point Oj ̸ = Oi do
19: Calculate distance ∥Oi − Oj ∥
20: end for
21: Determine q-distance  ̃disq
i for Oi
22: Find neighbors N eii within  ̃disq
i
23: reai,j ← max{  ̃disq
i , ∥Oi − Oj ∥} for Oj ∈ N eii ▷
Compute reachability distance
24: lrdi ← |N eii|∑
j∈N eii reai,j ▷ Calculate local reachability
density
25: lofi ←
∑
j∈N eii
lrdj
lrdi
|N eii|
26: end for
27: Umal, Unor ← {i | lofi > 1}, {i | lofi ≤ 1}
28: Step 4: Gradients Aggregation
29: for Client i ∈ Unor do
30: cre′
i ← tanh(crei) ▷ Compute adjusted confidence
31: wi ← cre′
i∑
i∈Unor cre′
i
▷ Normalize weights
32: end for
33: Output: θt
g ← θt−1
g + ∑
i∈Unor wi · ▽θt
i ▷ Aggregated
global model update
t-th round be denoted as θt, and the model of client i after
training in the t-th round be denoted as θt
i . Client i computes
its update in the t-th round as the difference between the
trained model θt
i and the global model received before training
θt, i.e., ▽θt
i = θt
i − θt. The client uploads the update ▽θt
i to
the central server. While aggregating, the central server records
the flattened result of each client’s update in the current round,
 ̄θt
i , as well as the global model θt−1 distributed by the central
server in the previous round. The central server retains the
models and update histories for up to T rounds.
2) Construction of attack intention: This problem can be
abstracted as rays in a high-dimensional space with specific
starting points. The starting point of a ray represents the
global model from the previous round, and the direction of
the ray represents the flattened update in the current round.
The optimization goal of the problem is to find a minimum
hypersphere that covers at least a ζ proportion of the rays. We
use the symbol Oi to represent the center of the hypersphere
for client i in the high-dimensional space and ri to represent
the radius of this hypersphere. Suppose the current round is t,
and t′ is one of the recent T rounds. Let  ̃Ot′
i denote the closest
point from the hypersphere center to the ray corresponding to
round t′. Then, the optimization problem can be defined as:
min
Oi,ri
ri (8)
s.t.
∣
∣
∣
{
t′ | ∥Oi −  ̃Ot′
i ∥ ≤ ri
}∣
∣
∣ ≥ ζT, (9)
max{t − T + 1, 1} ≤ t′ ≤ t. (10)
The main algorithm process for ”Obtain the Purpose Inten-
tion” is shown in lines 14-18 of Algorithm 2. This sec-
tion can be further divided into two parts: ”Construct Ray
Model” and ”Minimum Enclosing Hypersphere Calculation.”
The main process of ”Construct Ray Model” is as follows.
For each client Ri, the central server retains the global
model history and the update history for the most recent
T rounds. Let θt−T
g , θt−T +1
g , . . . , θt−1
g represent the global
models of the most recent T rounds, which have been ”flat-
tened” to a point in a high-dimensional space. Similarly, let
▽θt−T
i , ▽θt−T +1
i , . . . , ▽θt−1
i represent the flattened gradient
update vectors uploaded by client i in the most recent T
rounds. For each round t′ (max{t − T + 1, 1} ≤ t′ ≤ t), the
starting point of each ray is represented by the flattened global
model θt′
g , and the direction vector of each ray is represented
by the flattened gradient update ▽θt′
i . Thus, the ray model can
be constructed as follows:
{  ̃θt′−1
i = F latten(θt′ −1
g )
vt′
i = ▽θt′
i
, (11)
where  ̃θt′ −1
i is the starting point of the ray, and vt′
i is the
direction vector of the ray. The ray equation is then given by:
lt′
i =  ̃θt′ −1
i + αvt′
i , (12)
where α ∈ [0, +∞) is the parameter of the ray. The main
process of ”Minimum Enclosing Hypersphere Calculation” is
as follows. Based on the constructed ray model, the Minimum
Enclosing Hypersphere Ball (MEHB) algorithm is used to find
the smallest hypersphere that can cover at least a ζ proportion
of the rays. First, the initial center of the hypersphere Oi,0 is
set to the geometric center of all ray starting points, calculated
as follows:
Oi,0 = 1
min{T, t}
t∑
t′=max{t−T +1,1}
 ̃θt′ −1
g . (13)
The initial radius ri,0 is set as the maximum distance from
the initial center Oi,0 to all ray starting points:
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 7
ri,0 = t
max
t′ =max{t−T +1,1} ∥Oi,0 −  ̃θt′ −1
g ∥. (14)
Then, the center and radius are iteratively updated. For each
iteration k, compute the nearest points  ̃Ot′
i,k from the current
center Oi,k to all rays. For the t′-th round of each node Ri
(max{t−T +1, 1} ≤ t′ ≤ t), for the ray lt′
i , find the projection
point ˆOt′
i,k of the center Oi,k onto the current ray, and compute
the parameter αk:
αk = (Oi,k −  ̃θt′ −1
i ) · vt′
i
vt′
i · vt′
i
, (15)
Thus, the nearest point is given by  ̃Ot′
i,k =  ̃θt′ −1
i +
max {0, αk} vt′
i . Calculate the set of distances from the cur-
rent center Oi,k to these nearest points  ̃Ot′
i,k, denoted as
Disi,k = {∥Oi,k −  ̃Ot′
i,k∥, max{t − T + 1, 1} ≤ t′ ≤ t}. Sort
these distances in ascending order and temporarily discard a
proportion of 1−ζ of the rays to ignore the impact of outliers,
resulting in the set Dis′
i,k. The rounds corresponding to the
retained rays are denoted as  ̃T . Move the center towards the
farthest point  ̃Ot′
max
i,k among the retained nearest points:
Oi,k+1 = Oi,k + η′(  ̃Ot′
max
i,k − Oi,k), (16)
where η′ is the learning rate.
The radius of the moved center is:
ri,k+1 = {max{∥Oi,k+1 −  ̃Ot′
i,k+1∥}, t′ ∈  ̃T }, (17)
where  ̃Ot′
i,k+1 is the nearest point from the updated center
Oi,k+1 to the retained rays. When the difference between the
updated radius ri,k+1 and the previous radius ri,k is smaller
than a predefined threshold λ, or when the maximum number
of iterations kmax is reached, the iteration stops.
3) Abnormal Detection: The main process of ”Abnormal
Detection” identifies outliers based on the Local Outlier Fac-
tor (LOF). The LOF algorithm is a density-based anomaly
detection method. Its core idea is to calculate a value lofi to
reflect the degree of abnormality of node i. The value of lofi
represents the ratio of the average density of the intention
points around the intention point Oi to the density at the
location of Oi. The greater this ratio is than 1, the lower
the density around Oi compared to the surrounding intention
points, indicating that Oi is more likely to be an outlier.
First, we define the q-distance  ̃disq
i of the intention point Oi.
Assume there is an intention point Oj for node j in the high-
dimensional space, and the distance between Oj and Oi is
∥Oi − Oj ∥. If the following two conditions are satisfied, we
have  ̃disq
i = ∥Oi − Oj ∥:
• In the sample space, there are at least q intention points
O′
j such that ∥Oi − O′
j ∥ ≤ ∥Oi − Oj ∥, where j′ ̸ = i;
• In the sample space, there are at most q − 1 intention
points O′
j such that ∥Oi − O′
j ∥ < ∥Oi − Oj ∥, where
j′ ̸ = i.
In summary, the q-distance  ̃disq
i of Oi represents the distance
to the q-th farthest point in the high-dimensional space. Then,
we define the q-distance neighborhood N eii of the intention
point Oi as the set of all intention points whose distance to
Oi does not exceed  ̃disq
i . Since multiple data points at the
q-distance may exist simultaneously, |N eii| ≥ q. It can be
imagined that the q-distance of intention points with greater
outlier degrees is often larger, while those with smaller outlier
degrees tend to have smaller q-distances. Next, we define the
reachable distance of intention point Oi relative to intention
point Oj :
reai,j = max{  ̃disq
i , ∥Oi − Oj ∥}, (18)
That is, if the intention point Oj is far from the intention
point Oi, the reachable distance between them is their actual
distance ∥Oi − Oj ∥; if they are close enough, the reachable
distance is replaced by the q-distance  ̃disq
i of the intention
point Oi. Next, we define the local reachable density lrdi
of the intention point Oi as the reciprocal of the average
reachable distance of all intention points in its N eii:
lrdi = 1/(
∑
j∈N eii reai,j
|N eii| ) (19)
At this point, if there are duplicate points, lrd may become
infinite. lrdi can be understood as the density at the location
of intention point Oi. The higher the density, the more likely
intention point Oi belongs to the same cluster; the lower the
density, the more likely Oi is an outlier. In other words, if
intention point Oi and its surrounding neighborhood points
are in the same cluster, the reachable distance is likely to
be the smaller  ̃disq
i , resulting in a smaller sum of reachable
distances and a higher density value; if Oi is far from its
neighboring intention points, the reachable distance is likely
to take the larger ∥Oi − Oj ∥, resulting in a larger sum of
reachable distances and a lower density value, making it more
likely to be an outlier. Finally, we define the local outlier factor
lofi of intention point Oi as the average of the ratio of the
local reachable densities of all intention points in its N eii to
its own local reachable density:
lofi =
∑
j∈N eii
lrdj
lrdi
|N eii| =
∑
j∈N eii lrdj
|N eii| · lrdi
(20)
If lofi is close to 1, it indicates that the density of intention
point Oi is similar to its neighboring points, suggesting that
Oi and its neighborhood belong to the same cluster. If lofi
is less than 1, it means that the density of intention point
Oi is higher than its neighboring points, indicating that Oi
is a dense point. If lofi is greater than 1, it means that the
density of intention point Oi is lower than its neighboring
points, suggesting that Oi may be an outlier. In conclusion,
the LOF algorithm mainly determines whether Oi is an outlier
by comparing the density of each intention point Oi with that
of its neighboring points. The lower the density, the more
likely it is an outlier. Density is primarily calculated based on
the distance between points: the greater the distance between
points, the lower the density; the closer the distance, the higher
the density. After calculating the local outlier factor lofi for
each node i, clients with lofi greater than 1 are regarded as
abnormal clients, and their gradients are discarded
4) Gradients Aggregation: After removing abnormal nodes,
the aggregation is performed based on the confidence cre of
normal nodes. The definition is as follows:
crei = 1
ri + ρ , (21)
where ρ is a small positive number to prevent the denomi-
nator from being zero.
As shown in Figure 2, the LOF algorithm can identify
and remove potential malicious clients, but among the normal
nodes, there are still some nodes with very low confidence.
The minimum enclosing hypersphere of these nodes may
even overlap with the abnormal nodes. Therefore, in the
aggregation process, nodes with lower confidence should have
lower weights.
Consider the following scenario: suppose two nodes have
very small minimum enclosing hypersphere radii, but their
radii differ by a significant factor. Consequently, their com-
puted confidences would also differ significantly. However,
their intention points are quite clear, so their weights should
both be relatively high and not differ too much. Therefore, we
can use the Tanh activation function to adjust the confidence
weights:
cre′
i = tanh(crei), (22)
The Tanh activation function is a monotonic increasing func-
tion that rises quickly at first and then slowly within the range
of 0 to ∞. When crei → 0, cer′
i → 0, which means that
when the minimum enclosing hypersphere radius ri of node
i is large, its weight during aggregation is small. Conversely,
when crei → ∞, cer′
i → 1, which indicates that when the
radius ri is small, the weight of node i in the aggregation is
larger. This allows us to normalize each adjusted confidence
cer′
i to obtain the weight wi. The formula is:
wi = cre′
i
∑N
i=1 cre′
i
, (23)
Thus, the gradient aggregation formula for the model in the
t-th round is:
θt
g = θt−1
g +
N∑
i=1
wi▽θt
i , (24)
As shown in Figure 2, the diagram illustrates the computa-
tional process of anomaly detection and gradient aggregation.
Each sphere in the figure represents the intention range of a
specific node. The blue spheres represent benign nodes, while
the red spheres represent anomalous nodes. During anomaly
detection, we only consider the intention point of each node,
which is represented by the center of each sphere in the figure.
When calculating the reachable distance, since intention point
Oj1 is close to intention point Oi, the reachable distance
between them is the q-distance  ̃disq
i of intention point Oi.
Since intention point Oj2 is far from Oi, the reachable distance
between them is the actual distance ∥Oi −Oj2 ∥ between them.
The density of intention point Oi is higher than that of its
neighboring intention points, so lofi < 1. On the other hand,
Fig. 2. Benign and Malicious Nodes Before Aggregation
the density of intention point Oj2 is lower than that of its
neighboring intention points, so lofj2 > 1.
After removing anomalous nodes, the gradients of benign
nodes are aggregated with a weighting based on the confidence
of the normal nodes. As shown in the figure, although node
j1 is classified as a benign node, its confidence is very low,
and its hypersphere even overlaps with the anomalous node
j2. Therefore, its weight is small during gradient aggregation.
We summarize the algorithm in the form of pseudocode,
as shown in Algorithm 2. Lines 3 to 7 correspond to Step 1,
which outlines the method for retaining historical records for
T rounds. Lines 8 to 15 correspond to Step 2, which describes
the use of the minimum enclosing hypersphere algorithm to
determine each robot node’s intent point and confidence level.
Lines 16 to 27 correspond to Step 3, which explains how to
identify normal and abnormal nodes using the LOF algorithm.
Finally, lines 28 to 32 correspond to Step 4, which illustrates
the method for performing weighted aggregation based on
each robot node’s confidence level.
```

据此写一个函数，首先通过最小覆盖球算法获取恶意客户端意图点，再通过lof聚类，返回恶意客户端的结果，并返回lof分作为热力图依据。

可以参考demo：

```
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor

# 第二步: 获取意图点 (Obtain the Purpose Intention)
def obtain_purpose_intention(clients_data, T, zeta, eta_prime, max_iter, lambda_thresh):
    client_intentions = {}
    
    for client_id, data in clients_data.items():
        theta_history = data['theta_history']  # T轮次的展平全局模型
        grad_updates = data['grad_updates']  # T轮次的展平梯度更新
        
        O_i = np.mean(theta_history, axis=0)
        r_i = max(np.linalg.norm(O_i - theta_history, axis=1))
        
        for k in range(max_iter):
            projection_points = []
            for t in range(T):
                theta_t = theta_history[t]
                v_t = grad_updates[t]
                alpha_k = max(0, np.dot(O_i - theta_t, v_t) / np.dot(v_t, v_t))
                proj_point = theta_t + alpha_k * v_t
                projection_points.append(proj_point)
            
            distances = np.linalg.norm(O_i - np.array(projection_points), axis=1)
            selected_indices = np.argsort(distances)[:int(zeta * T)]
            max_proj_point = projection_points[selected_indices[-1]]
            
            O_i = O_i + eta_prime * (max_proj_point - O_i)
            r_new = max(np.linalg.norm(O_i - np.array(projection_points)[selected_indices], axis=1))
            
            if abs(r_new - r_i) < lambda_thresh:
                break
            
            r_i = r_new
        
        client_intentions[client_id] = {'center': O_i, 'radius': r_i}
    
    return client_intentions

# 第三步: 异常检测 (Abnormal Detection)
def abnormal_detection(client_intentions, k_neighbors=5):
    intention_points = np.array([v['center'] for v in client_intentions.values()])
    lof = LocalOutlierFactor(n_neighbors=k_neighbors)
    lof_scores = lof.fit_predict(intention_points)
    
    U_mal = [client_id for i, client_id in enumerate(client_intentions) if lof_scores[i] == -1]
    U_nor = [client_id for i, client_id in enumerate(client_intentions) if lof_scores[i] != -1]
    
    return U_mal, U_nor, intention_points, lof_scores

# 可视化意图点并标注异常点
def plot_intentions(intention_points, lof_scores, filename='intentions_plot.svg'):
    # 使用PCA将高维数据降到二维
    pca = PCA(n_components=2)
    points_2d = pca.fit_transform(intention_points)
    
    # 绘图
    plt.figure(figsize=(10, 7))
    for i, point in enumerate(points_2d):
        if lof_scores[i] == -1:  # 异常点
            plt.scatter(point[0], point[1], color='r', marker='x', label='Anomalous Client' if i == 0 else "")
        else:  # 正常点
            plt.scatter(point[0], point[1], color='b', marker='o', label='Normal Client' if i == 0 else "")
    
    plt.title('Visualization of Client Intentions and Anomalies')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    
    # 保存为SVG格式
    plt.savefig(filename, format='svg')
    plt.show()

# 生成模拟数据，包含异常用户
def generate_simulated_data(num_clients=20, num_features=100, T=10, anomaly_ratio=0.15):
    clients_data = {}
    np.random.seed(42)  # 固定随机种子，确保每次运行的结果相同
    num_anomalies = int(num_clients * anomaly_ratio)  # 确定异常用户的数量

    for i in range(num_clients):
        # 模拟生成全局模型的历史记录和梯度更新
        theta_history = np.random.rand(T, num_features)
        grad_updates = np.random.rand(T, num_features)

        # 为异常用户添加噪声或偏移
        if i < num_anomalies:
            theta_history += np.random.normal(5, 1, size=theta_history.shape)  # 添加偏移
            grad_updates += np.random.normal(5, 1, size=grad_updates.shape)  # 添加偏移
        
        clients_data[f'client{i+1}'] = {'theta_history': theta_history, 'grad_updates': grad_updates}
    
    return clients_data

# 生成模拟数据，设置约15%的异常用户
clients_data = generate_simulated_data(num_clients=20, num_features=100, T=10, anomaly_ratio=0.15)

# 获取意图点
client_intentions = obtain_purpose_intention(clients_data, T=10, zeta=0.8, eta_prime=0.1, max_iter=100, lambda_thresh=1e-4)

# 异常检测
U_mal, U_nor, intention_points, lof_scores = abnormal_detection(client_intentions, k_neighbors=5)

# 绘制结果并保存为SVG
plot_intentions(intention_points, lof_scores, filename='intentions_plot.svg')

```






你不需要可视化，你只需要写一个函数，根据`calc_maliciousAndCos_justByGrads`中同类型的梯度进行识别，返回恶意客户端的结果，并返回lof分数
```
# 通过历史记录的梯度计算恶意客户端
def calc_maliciousAndCos_justByGrads(roundsNum: List[int], dirPath: str):
    clientPerRound = 10  # 这里就先写死了
    maliciousPerRound = 3
    gradients = [0] * len(roundsNum) * clientPerRound
    maliciouses = []
    for th, roundNum in enumerate(roundsNum):
        participants_thisRound = getParticipants(roundNum, dirPath)
        model_updates = loadPkl(roundNum, dirPath)
        keys = list(model_updates.keys())
        last_layer_updates = model_updates[keys[-2]]
        K = len(last_layer_updates)
        for i in range(K):
            thisParticipant = participants_thisRound[i]
            if thisParticipant < maliciousPerRound:
                thisIndex = th * maliciousPerRound + thisParticipant
            else:
                thisIndex = maliciousPerRound * len(roundsNum) + (clientPerRound - maliciousPerRound) * th + thisParticipant - maliciousPerRound
            gradients[thisIndex] = last_layer_updates[i].cpu().numpy()
        # _, foolsgoldMaliciousIndex, _ = kmeans_clustering(gradients)
        _, foolsgoldMaliciousIndex, _ = foolsgold_oneRound(roundNum, dirPath)
        for thisMaliciousIndex in foolsgoldMaliciousIndex:
            if thisMaliciousIndex < maliciousPerRound:
                thisIndex = th * maliciousPerRound + thisMaliciousIndex
            else:
                thisIndex = maliciousPerRound * len(roundsNum) + (clientPerRound - maliciousPerRound) * th + thisMaliciousIndex - maliciousPerRound
            maliciouses.append(thisIndex)
    shuffleArray = np.arange(50)
    np.random.shuffle(shuffleArray[0:len(roundsNum) * maliciousPerRound])
    np.random.shuffle(shuffleArray[len(roundsNum) * maliciousPerRound:50])
    malicious_shuffled = [shuffleArray[i] for i in maliciouses]
    gradients_shuffled = [gradients[shuffleArray[i]] for i in range(50)]
    cs = smp.cosine_similarity(gradients_shuffled)
    cs_tensor = torch.from_numpy(cs)
    return malicious_shuffled, cs_tensor
```




不需要`获取最后一层更新`，但需要体现“球心”、意图点






解释这段代码
```
def calc_maliciousAndCos_justByGrads(roundsNum: List[int], dirPath: str):
    clientPerRound = 10  # 这里就先写死了
    maliciousPerRound = 3
    gradients = [0] * len(roundsNum) * clientPerRound
    maliciouses = []
    for th, roundNum in enumerate(roundsNum):
        participants_thisRound = getParticipants(roundNum, dirPath)
        model_updates = loadPkl(roundNum, dirPath)
        keys = list(model_updates.keys())
        last_layer_updates = model_updates[keys[-2]]
        K = len(last_layer_updates)
        for i in range(K):
            thisParticipant = participants_thisRound[i]
            if thisParticipant < maliciousPerRound:
                thisIndex = th * maliciousPerRound + thisParticipant
            else:
                thisIndex = maliciousPerRound * len(roundsNum) + (clientPerRound - maliciousPerRound) * th + thisParticipant - maliciousPerRound
            gradients[thisIndex] = last_layer_updates[i].cpu().numpy()
        # _, foolsgoldMaliciousIndex, _ = kmeans_clustering(gradients)
        _, foolsgoldMaliciousIndex, _ = foolsgold_oneRound(roundNum, dirPath)
        for thisMaliciousIndex in foolsgoldMaliciousIndex:
            if thisMaliciousIndex < maliciousPerRound:
                thisIndex = th * maliciousPerRound + thisMaliciousIndex
            else:
                thisIndex = maliciousPerRound * len(roundsNum) + (clientPerRound - maliciousPerRound) * th + thisMaliciousIndex - maliciousPerRound
            maliciouses.append(thisIndex)
    shuffleArray = np.arange(50)
    np.random.shuffle(shuffleArray[0:len(roundsNum) * maliciousPerRound])
    np.random.shuffle(shuffleArray[len(roundsNum) * maliciousPerRound:50])
    malicious_shuffled = [shuffleArray[i] for i in maliciouses]
    gradients_shuffled = [gradients[shuffleArray[i]] for i in range(50)]
    cs = smp.cosine_similarity(gradients_shuffled)
    cs_tensor = torch.from_numpy(cs)
    return malicious_shuffled, cs_tensor
```







```
def calc_maliciousAndCos_justByGrads_SecFFT(roundsNum: List[int], dirPath: str):
    clientPerRound = 10  # 这里就先写死了
    maliciousPerRound = 3
    gradients = [0] * len(roundsNum) * clientPerRound
    maliciouses = []
    for th, roundNum in enumerate(roundsNum):
        participants_thisRound = getParticipants(roundNum, dirPath)
        model_updates = loadPkl(roundNum, dirPath)
        keys = list(model_updates.keys())
        last_layer_updates = model_updates[keys[-2]]
        K = len(last_layer_updates)
        for i in range(K):
            thisParticipant = participants_thisRound[i]
            if thisParticipant < maliciousPerRound:
                thisIndex = th * maliciousPerRound + thisParticipant
            else:
                thisIndex = maliciousPerRound * len(roundsNum) + (clientPerRound - maliciousPerRound) * th + thisParticipant - maliciousPerRound
            gradients[thisIndex] = last_layer_updates[i].cpu().numpy()
        # _, foolsgoldMaliciousIndex, _ = kmeans_clustering(gradients)
        _, foolsgoldMaliciousIndex, _ = foolsgold_oneRound(roundNum, dirPath)
        for thisMaliciousIndex in foolsgoldMaliciousIndex:
            if thisMaliciousIndex < maliciousPerRound:
                thisIndex = th * maliciousPerRound + thisMaliciousIndex
            else:
                thisIndex = maliciousPerRound * len(roundsNum) + (clientPerRound - maliciousPerRound) * th + thisMaliciousIndex - maliciousPerRound
            maliciouses.append(thisIndex)
    shuffleArray = np.arange(50)
    np.random.shuffle(shuffleArray[0:len(roundsNum) * maliciousPerRound])
    np.random.shuffle(shuffleArray[len(roundsNum) * maliciousPerRound:50])
    malicious_shuffled = [shuffleArray[i] for i in maliciouses]
    gradients_shuffled = [gradients[shuffleArray[i]] for i in range(50)]
    # TODO: 在这里写最小覆盖球算法和LOF，将gradients_shuffled作为输入
    cs = smp.cosine_similarity(gradients_shuffled)
    cs_tensor = torch.from_numpy(cs)
    return malicious_shuffled, cs_tensor
```




计算意图点那里再写地复杂一点，注意是通过最小覆盖超球算法求意图点




注意是最小覆盖超球，是多维的。





请使用论文中通过迭代移动球心位置的办法






```
def min_enclosing_ball(points, zeta, eta_prime, max_iter, lambda_thresh):
    """
    使用迭代方法计算最小覆盖超球的球心和半径。
    
    参数:
    - points: 客户端的历史梯度更新，形状为 (T, d)
    - zeta: 覆盖比例
    - eta_prime: 学习率
    - max_iter: 最大迭代次数
    - lambda_thresh: 收敛阈值
    
    返回:
    - center: 球心（意图点）
    - radius: 球半径
    """
    # 确保 points 是二维数组
    points = np.array(points)
    if len(points.shape) == 1:
        points = points.reshape(1, -1)
    
    # 初始化球心为所有点的均值
    O_i = np.mean(points, axis=0)
    # 初始化球半径为当前球心到所有点的最大距离
    r_i = max(np.linalg.norm(O_i - points, axis=1))
    
    # 迭代优化球心和半径
    for k in range(max_iter):
        projection_points = []
        for point in points:
            # 计算每个点到球心的投影点
            direction = O_i - point
            norm_direction = np.linalg.norm(direction)
            if norm_direction == 0:
                continue
            alpha_k = max(0, np.dot(O_i - point, direction) / np.dot(direction, direction))
            proj_point = point + alpha_k * direction
            projection_points.append(proj_point)
        
        # 计算投影点到球心的距离
        distances = np.linalg.norm(O_i - np.array(projection_points), axis=1)
        # 选择前 zeta*T 个投影点
        selected_indices = np.argsort(distances)[:int(zeta * len(points))]
        max_proj_point = np.array(projection_points)[selected_indices[-1]]  # 最远的投影点
        
        # 更新球心
        O_i = O_i + eta_prime * (max_proj_point - O_i)
        # 计算新的半径
        r_new = max(np.linalg.norm(O_i - np.array(projection_points)[selected_indices], axis=1))
        
        # 检查收敛条件
        if abs(r_new - r_i) < lambda_thresh:
            break
        
        r_i = r_new
    
    return O_i, r_i

def calc_maliciousAndCos_justByGrads_SecFFT(roundsNum: List[int], dirPath: str, zeta=0.8, eta_prime=0.1, max_iter=100, lambda_thresh=1e-4):
    clientPerRound = 10  # 这里就先写死了
    maliciousPerRound = 3
    gradients = [0] * len(roundsNum) * clientPerRound
    maliciouses = []
    for th, roundNum in enumerate(roundsNum):
        participants_thisRound = getParticipants(roundNum, dirPath)
        model_updates = loadPkl(roundNum, dirPath)
        keys = list(model_updates.keys())
        last_layer_updates = model_updates[keys[-2]]
        K = len(last_layer_updates)
        for i in range(K):
            thisParticipant = participants_thisRound[i]
            if thisParticipant < maliciousPerRound:
                thisIndex = th * maliciousPerRound + thisParticipant
            else:
                thisIndex = maliciousPerRound * len(roundsNum) + (clientPerRound - maliciousPerRound) * th + thisParticipant - maliciousPerRound
            gradients[thisIndex] = last_layer_updates[i].cpu().numpy()
        # _, foolsgoldMaliciousIndex, _ = kmeans_clustering(gradients)
        _, foolsgoldMaliciousIndex, _ = foolsgold_oneRound(roundNum, dirPath)
        for thisMaliciousIndex in foolsgoldMaliciousIndex:
            if thisMaliciousIndex < maliciousPerRound:
                thisIndex = th * maliciousPerRound + thisMaliciousIndex
            else:
                thisIndex = maliciousPerRound * len(roundsNum) + (clientPerRound - maliciousPerRound) * th + thisMaliciousIndex - maliciousPerRound
            maliciouses.append(thisIndex)
    shuffleArray = np.arange(50)
    np.random.shuffle(shuffleArray[0:len(roundsNum) * maliciousPerRound])
    np.random.shuffle(shuffleArray[len(roundsNum) * maliciousPerRound:50])
    malicious_shuffled = [shuffleArray[i] for i in maliciouses]
    gradients_shuffled = [gradients[shuffleArray[i]] for i in range(50)]
    # TODO: 在这里写最小覆盖球算法和LOF，将gradients_shuffled作为输入
    intentions = []
    for grad in gradients_shuffled:
        grad = np.array(grad)
        # 使用最小覆盖球算法计算意图点
        center, radius = min_enclosing_ball(grad, zeta, eta_prime, max_iter, lambda_thresh)
        intentions.append(center)  # 意图点是球心
    lof = LocalOutlierFactor(n_neighbors=5, metric='euclidean')
    lof_scores = lof.fit_predict(intentions)  # 使用LOF检测异常客户端
    detected_malicious = [i for i, score in enumerate(lof_scores) if score == -1]
    print(detected_malicious)
    cs = smp.cosine_similarity(gradients_shuffled)
    cs_tensor = torch.from_numpy(cs)
    return malicious_shuffled, cs_tensor
```
```
# malicious_clients, cosine_similarity_matrix = fltrust([15, 16, 17, 18, 19], '../NormalRun/FL_Backdoor_CV/resultWithTime/2024-09-14_15-11-15-fltrust-fmnist_NEUROTOXIN', '../NormalRun/FL_Backdoor_CV/saved_models/Revision_1/fltrust_NEUROTOXIN_09141511-fmnist')
# foolsgoldMaliciousIndex, foolsgoldScore2 = calc_maliciousAndCos_justByGrads([15, 16, 17, 18, 19], '../NormalRun/FL_Backdoor_CV/resultWithTime/2024-09-13_23-15-48-foolsgold-fmnist_NEUROTOXIN')
# fltrustMaliciousIndex, fltrustScore2 = calc_maliciousAndCos_justByGrads([15, 16, 17, 18, 19], '../NormalRun/FL_Backdoor_CV/resultWithTime/2024-09-14_15-11-15-fltrust-fmnist_NEUROTOXIN')
# # flameMaliciousIndex, flameMaliciousIndex = calc_maliciousAndCos_justByGrads([15, 16, 17, 18, 19], '../NormalRun/FL_Backdoor_CV/resultWithTime/2024-09-12_23-30-57-flame-fmnist_NEUROTOXIN')
# flameMaliciousIndex, flameScore2 = calc_maliciousAndCos_justByGrads([10, 11, 12, 13, 14], '../NormalRun/FL_Backdoor_CV/resultWithTime/2024-09-14_15-11-15-fltrust-fmnist_NEUROTOXIN')
secfftMaliciousIndex, secfftScore2 = calc_maliciousAndCos_justByGrads_SecFFT([10, 11, 12, 13, 14], '../NormalRun/FL_Backdoor_CV/resultWithTime/2024-09-14_15-11-15-fltrust-fmnist_NEUROTOXIN')  # 随便找个历史保存的数据进行识别

# foolsgoldScore2 = calc_maliciousAndCos_justByGrads([15, 16, 17, 18, 19], '../NormalRun/FL_Backdoor_CV/resultWithTime/2024-09-13_23-15-48-foolsgold-fmnist_NEUROTOXIN')

# print(foolsgoldMaliciousIndex)
# print(foolsgoldScore2)
# datas = [foolsgoldScore2] * 12
datas = [secfftScore2] * 12
# datas = [foolsgoldScore2, fltrustScore2, flameScore2, secfftScore2] * 3
# print(foolsgoldMaliciousIndex)
# print(fltrustMaliciousIndex)
# print(flameMaliciousIndex)
plot_detection_heatmaps_3x4(*datas)
```




不报错了，但识别结果不好。`intentions`结果如下：
```
[array([ 0.00265263, -0.00137249,  0.00249858, ...,  0.00259138,
       -0.00192236,  0.00262321], dtype=float32), array([ 0.00266748, -0.00238248,  0.0026174 , ...,  0.00263826,
       -0.00226646,  0.00264889], dtype=float32), array([ 0.00266336, -0.0024231 ,  0.00265443, ...,  0.00259403,
       -0.00240742,  0.00264275], dtype=float32), array([ 0.00266584, -0.00215233,  0.00257359, ...,  0.00248383,
       -0.00184622,  0.00265152], dtype=float32), array([ 0.00266422, -0.00244031,  0.00256623, ...,  0.00257299,
       -0.00240531,  0.0025933 ], dtype=float32), array([ 0.00266855, -0.00253644,  0.0026398 , ...,  0.00265534,
       -0.00253061,  0.00266118], dtype=float32), array([ 0.002666  , -0.00265254,  0.00264   , ...,  0.00264987,
       -0.00246943,  0.00266074], dtype=float32), array([ 0.00265845, -0.00249436,  0.00266516, ...,  0.00259749,
       -0.00236242,  0.00263929], dtype=float32), array([ 0.00266142, -0.00211681,  0.00266169, ...,  0.00257311,
       -0.00190277,  0.0026312 ], dtype=float32), array([ 0.00266819, -0.00215402,  0.00264693, ...,  0.00255591,
       -0.0021921 ,  0.0026177 ], dtype=float32), array([ 0.00266377, -0.0018989 ,  0.00264109, ...,  0.00250322,
       -0.00200094,  0.00256392], dtype=float32), array([ 0.00267051, -0.00203082,  0.00257989, ...,  0.00253621,
       -0.00196387,  0.00260156], dtype=float32), array([ 0.00265029, -0.00242873,  0.00256764, ...,  0.00265834,
       -0.00233002,  0.00262154], dtype=float32), array([ 0.00266901, -0.00240869,  0.00263937, ...,  0.002508  ,
       -0.00176334,  0.00264929], dtype=float32), array([ 0.00267088, -0.00249366,  0.00264148, ...,  0.00263755,
       -0.00225316,  0.00262441], dtype=float32), array([ 0.00049924, -0.00042198,  0.00049536, ...,  0.00049483,
       -0.00046745,  0.00049805], dtype=float32), array([ 0.00049841, -0.00031816,  0.00048016, ...,  0.00042054,
       -0.0002236 ,  0.00048557], dtype=float32), array([ 0.0004993 , -0.0004779 ,  0.00046018, ...,  0.00047225,
       -0.00035145,  0.00049374], dtype=float32), array([ 0.00049975, -0.00044948,  0.00049246, ...,  0.00047098,
       -0.00042914,  0.00049535], dtype=float32), array([ 0.00049815, -0.00038285,  0.00048942, ...,  0.00047652,
       -0.00039652,  0.00049494], dtype=float32), array([ 0.00049961, -0.00023054,  0.00045384, ...,  0.00047554,
       -0.00038248,  0.00049227], dtype=float32), array([ 0.00049976, -0.00046681,  0.00049576, ...,  0.00049565,
       -0.00046569,  0.00049655], dtype=float32), array([ 0.00049983, -0.00021165,  0.0004404 , ...,  0.000458  ,
       -0.00040009,  0.00049095], dtype=float32), array([ 0.00049859, -0.00049883,  0.00047823, ...,  0.00049491,
       -0.00047786,  0.00049895], dtype=float32), array([ 0.00049882, -0.00042841,  0.00048307, ...,  0.00048717,
       -0.00046529,  0.0004989 ], dtype=float32), array([ 0.00049978, -0.00046772,  0.00047827, ...,  0.0004492 ,
       -0.00034373,  0.00047295], dtype=float32), array([ 0.00049939, -0.00046792,  0.00040958, ...,  0.00047972,
       -0.00040275,  0.00049889], dtype=float32), array([ 0.00049991, -0.0004514 ,  0.00048983, ...,  0.00049759,
       -0.00048522,  0.00049494], dtype=float32), array([ 0.00050094, -0.00046453,  0.00047546, ...,  0.00047742,
       -0.00039666,  0.00048594], dtype=float32), array([ 0.00049913, -0.00042558,  0.000484  , ...,  0.00049071,
       -0.00045225,  0.00049785], dtype=float32), array([ 0.00049803, -0.00046871,  0.00047976, ...,  0.00048038,
       -0.00043405,  0.00049309], dtype=float32), array([ 0.00049831, -0.00041612,  0.00049969, ...,  0.00046573,
       -0.00035113,  0.0004862 ], dtype=float32), array([ 0.00050041, -0.00040688,  0.00048896, ...,  0.00049721,
       -0.00048188,  0.00049832], dtype=float32), array([ 0.00049974, -0.00037836,  0.00048278, ...,  0.00047451,
       -0.00032065,  0.00048311], dtype=float32), array([ 0.00049973, -0.00043384,  0.00047782, ...,  0.00045229,
       -0.00019887,  0.00049106], dtype=float32), array([ 0.00049736, -0.00043529,  0.00050021, ...,  0.00048694,
       -0.00047854,  0.00049088], dtype=float32), array([ 0.00049841, -0.00045324,  0.00049347, ...,  0.00048904,
       -0.00045836,  0.00049303], dtype=float32), array([ 0.00050034, -0.00044064,  0.00047185, ...,  0.00049704,
       -0.00046119,  0.00049884], dtype=float32), array([ 0.00050069, -0.0004802 ,  0.00049495, ...,  0.00047824,
       -0.00020524,  0.00049601], dtype=float32), array([ 0.00049782, -0.0003997 ,  0.00041994, ...,  0.00045997,
       -0.00033355,  0.00048356], dtype=float32), array([ 0.00050079, -0.0004763 ,  0.00049539, ...,  0.00048869,
       -0.00045658,  0.00049262], dtype=float32), array([ 0.00049961, -0.00041601,  0.00048275, ...,  0.00041432,
       -0.00016217,  0.00044485], dtype=float32), array([ 0.00050012, -0.00038492,  0.00049186, ...,  0.00047177,
       -0.00036988,  0.00049353], dtype=float32), array([ 0.00050062, -0.00042228,  0.00047321, ...,  0.00046985,
       -0.00026206,  0.00047917], dtype=float32), array([ 0.00049827, -0.00039642,  0.0004771 , ...,  0.00048907,
       -0.00041465,  0.0004945 ], dtype=float32), array([ 0.00049776, -0.00033221,  0.00048208, ...,  0.00047757,
       -0.00039765,  0.00048814], dtype=float32), array([ 0.00049958, -0.00042632,  0.0004924 , ...,  0.00047343,
       -0.00037083,  0.00049589], dtype=float32), array([ 0.00049832, -0.00040325,  0.00042166, ...,  0.00046089,
       -0.00038474,  0.00049383], dtype=float32), array([ 0.00049996, -0.00041239,  0.00047179, ...,  0.00048409,
       -0.00027607,  0.00048417], dtype=float32), array([ 0.00049975, -0.00049731,  0.00049793, ...,  0.00047118,
       -0.00037749,  0.00049632], dtype=float32)]
```
其中前15个为恶意。
应该如何识别？






```
intentions = []
    for grad in gradients_shuffled:
        grad = np.array(grad)
        # 使用最小覆盖球算法计算意图点
        center, radius = min_enclosing_ball(grad, zeta, eta_prime, max_iter, lambda_thresh)
        intentions.append(center)  # 意图点是球心
    print(intentions)
    print(len(intentions))
    # 计算意图点的L2范数
    norms = np.linalg.norm(intentions, axis=1)

    # 设置阈值
    threshold = 0.001  # 根据数据进行调整
    detected_malicious = [i for i, norm in enumerate(norms) if norm > threshold]
    print(f'detected_malicious: {detected_malicious}')

    # 使用LOF算法进一步检测异常
    lof = LocalOutlierFactor(n_neighbors=5, metric='euclidean')
    lof_scores = lof.fit_predict(intentions)  # 使用LOF检测异常客户端
    detected_malicious_lof = [i for i, score in enumerate(lof_scores) if score == -1]
    print(f'detected_malicious_lof: {detected_malicious_lof}')

    # 合并两种方法的结果
    detected_malicious = list(set(detected_malicious + detected_malicious_lof))

    # 输出结果
    print("Detected malicious clients:", detected_malicious)
```
输出结果：
```
50
detected_malicious: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
detected_malicious_lof: []
Detected malicious clients: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
```

显然效果不好。
能否只根据意图点的第一个值（0.2  和  0.4）附近，调整lof的参数，以便更好的识别？





很棒！
但是，lof.fit_predict只能得到-1或者1，lof有没有一个函数，可以得到每个客户端的lof得分？