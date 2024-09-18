'''
Author: LetMeFly
Date: 2024-09-10 02:07:00
LastEditors: LetMeFly
LastEditTime: 2024-09-10 02:29:13
'''
import numpy as np
from sklearn.neighbors import LocalOutlierFactor

# 第二步: 获取意图点 (Obtain the Purpose Intention)
def obtain_purpose_intention(clients_data, T, zeta, eta_prime, max_iter, lambda_thresh):
    # 初始化每个客户端的意图点
    client_intentions = {}
    
    for client_id, data in clients_data.items():
        # 构建射线模型
        theta_history = data['theta_history']  # T轮次的展平全局模型
        grad_updates = data['grad_updates']  # T轮次的展平梯度更新
        
        # 初始球心为射线起点的几何中心
        O_i = np.mean(theta_history, axis=0)
        # 初始半径为球心到射线起点的最大距离
        r_i = max(np.linalg.norm(O_i - theta_history, axis=1))
        
        # 最小覆盖超球迭代
        for k in range(max_iter):
            projection_points = []
            # 计算每个射线的投影点
            for t in range(T):
                theta_t = theta_history[t]
                v_t = grad_updates[t]
                alpha_k = max(0, np.dot(O_i - theta_t, v_t) / np.dot(v_t, v_t))
                proj_point = theta_t + alpha_k * v_t
                projection_points.append(proj_point)
            
            # 计算所有投影点到球心的距离
            distances = np.linalg.norm(O_i - np.array(projection_points), axis=1)
            # 排序并选择覆盖ζ比例射线的点
            selected_indices = np.argsort(distances)[:int(zeta * T)]
            max_proj_point = projection_points[selected_indices[-1]]
            
            # 更新球心和半径
            O_i = O_i + eta_prime * (max_proj_point - O_i)
            r_new = max(np.linalg.norm(O_i - np.array(projection_points)[selected_indices], axis=1))
            
            # 检查收敛条件
            if abs(r_new - r_i) < lambda_thresh:
                break
            
            r_i = r_new
        
        client_intentions[client_id] = {'center': O_i, 'radius': r_i}
    
    return client_intentions

# 第三步: 异常检测 (Abnormal Detection)
def abnormal_detection(client_intentions, k_neighbors=5):
    # 提取所有客户端的意图点作为输入
    intention_points = np.array([v['center'] for v in client_intentions.values()])
    
    # 使用局部离群因子（LOF）进行异常检测
    lof = LocalOutlierFactor(n_neighbors=k_neighbors)
    lof_scores = lof.fit_predict(intention_points)
    
    # 根据LOF结果标记异常和正常客户端
    U_mal = [client_id for i, client_id in enumerate(client_intentions) if lof_scores[i] == -1]
    U_nor = [client_id for i, client_id in enumerate(client_intentions) if lof_scores[i] != -1]
    
    return U_mal, U_nor

# 第四步: 梯度聚合 (Gradients Aggregation)
def gradients_aggregation(clients_data, client_intentions, U_nor, rho=1e-5):
    # 计算正常客户端的置信度
    confidences = {client_id: 1 / (client_intentions[client_id]['radius'] + rho) for client_id in U_nor}
    
    # 使用tanh对置信度进行加权处理
    adjusted_confidences = {client_id: np.tanh(conf) for client_id, conf in confidences.items()}
    
    # 归一化权重
    total_weight = sum(adjusted_confidences.values())
    weights = {client_id: adjusted_confidences[client_id] / total_weight for client_id in U_nor}
    
    # 聚合正常客户端的梯度
    aggregated_gradient = np.zeros_like(clients_data[U_nor[0]]['grad_updates'][-1])  # 假设所有客户端的更新大小一致
    for client_id in U_nor:
        aggregated_gradient += weights[client_id] * clients_data[client_id]['grad_updates'][-1]
    
    return aggregated_gradient

# 示例数据（需替换为实际数据）
clients_data = {
    'client1': {'theta_history': np.random.rand(10, 100), 'grad_updates': np.random.rand(10, 100)},
    'client2': {'theta_history': np.random.rand(10, 100), 'grad_updates': np.random.rand(10, 100)},
    # 添加更多客户端数据...
}

# 运行第二步：获取意图点
client_intentions = obtain_purpose_intention(clients_data, T=10, zeta=0.8, eta_prime=0.1, max_iter=100, lambda_thresh=1e-4)

# 运行第三步：异常检测
U_mal, U_nor = abnormal_detection(client_intentions, k_neighbors=5)

# 运行第四步：梯度聚合
aggregated_gradient = gradients_aggregation(clients_data, client_intentions, U_nor)

print("Aggregated Gradient:", aggregated_gradient)
