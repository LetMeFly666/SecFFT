'''
Author: LetMeFly
Date: 2024-09-14 00:47:51
LastEditors: LetMeFly
LastEditTime: 2024-09-14 00:48:24
'''
from sklearn.metrics import matthews_corrcoef, roc_auc_score

# 假设共有20个客户端，前4个为恶意客户端
total_clients = 20
malicious_clients = {0, 1, 2, 3}  # 实际恶意客户端的索引

# 假设识别出来的恶意客户端结果（可以修改此数组来测试不同情况）
identified_malicious = [0, 1, 4, 5]  # 被识别为恶意的客户端索引

# 计算TP, FP, TN, FN
TP = len(malicious_clients.intersection(identified_malicious))
FP = len(set(identified_malicious) - malicious_clients)
FN = len(malicious_clients - set(identified_malicious))
TN = total_clients - TP - FP - FN

# 准确率（Accuracy）
accuracy = (TP + TN) / total_clients

# 精确率（Precision）
precision = TP / (TP + FP) if (TP + FP) > 0 else 0

# 召回率（Recall）
recall = TP / (TP + FN) if (TP + FN) > 0 else 0

# 特异度（Specificity）
specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

# 误报率（False Positive Rate, FPR）
fpr = FP / (FP + TN) if (FP + TN) > 0 else 0

# 假阴率（False Negative Rate, FNR）
fnr = FN / (TP + FN) if (TP + FN) > 0 else 0

# 计算AUC（此处假设预测的概率值，以便计算AUC）
y_true = [1 if i in malicious_clients else 0 for i in range(total_clients)]
y_scores = [1 if i in identified_malicious else 0 for i in range(total_clients)]
auc = roc_auc_score(y_true, y_scores)

# 计算MCC（Matthews Correlation Coefficient）
mcc = matthews_corrcoef(y_true, y_scores)

# 打印结果
print(f"TP (True Positive): {TP}")
print(f"FP (False Positive): {FP}")
print(f"FN (False Negative): {FN}")
print(f"TN (True Negative): {TN}")
print(f"准确率（Accuracy）: {accuracy:.2f}")
print(f"精确率（Precision）: {precision:.2f}")
print(f"召回率（Recall）: {recall:.2f}")
print(f"特异度（Specificity）: {specificity:.2f}")
print(f"误报率（False Positive Rate, FPR）: {fpr:.2f}")
print(f"假阴率（False Negative Rate, FNR）: {fnr:.2f}")
print(f"AUC（Area Under Curve）: {auc:.2f}")
print(f"MCC（Matthews Correlation Coefficient）: {mcc:.2f}")
