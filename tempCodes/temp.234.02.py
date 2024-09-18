'''
Author: LetMeFly
Date: 2024-09-10 02:34:55
LastEditors: LetMeFly
LastEditTime: 2024-09-10 02:34:59
'''
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
def plot_intentions(intention_points, lof_scores):
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
    plt.show()

# 示例数据（需替换为实际数据）
clients_data = {
    'client1': {'theta_history': np.random.rand(10, 100), 'grad_updates': np.random.rand(10, 100)},
    'client2': {'theta_history': np.random.rand(10, 100), 'grad_updates': np.random.rand(10, 100)},
    # 添加更多客户端数据...
}

# 获取意图点
client_intentions = obtain_purpose_intention(clients_data, T=10, zeta=0.8, eta_prime=0.1, max_iter=100, lambda_thresh=1e-4)

# 异常检测
U_mal, U_nor, intention_points, lof_scores = abnormal_detection(client_intentions, k_neighbors=5)

# 绘制结果
plot_intentions(intention_points, lof_scores)
