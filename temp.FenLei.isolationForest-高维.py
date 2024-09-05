'''
Author: LetMeFly
Date: 2024-09-04 10:33:08
LastEditors: LetMeFly
LastEditTime: 2024-09-04 10:37:39
'''
from sklearn.ensemble import IsolationForest
import numpy as np

# 创建一个5维数据集，每个点有5个坐标 (x1, x2, x3, x4, x5) 和一个置信度特征 (confidence)
np.random.seed(42)
X = np.random.randn(100, 6)  # 100个数据点，每个点有6个特征（5维坐标 + 1个置信度）

# 引入一些异常点，确保它们在高维空间中明显不同
outliers = np.random.uniform(low=-10, high=10, size=(10, 6))
X = np.vstack((X, outliers))

# 初始化IsolationForest模型
model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)

# 使用fit_predict方法进行训练和预测
results = model.fit_predict(X)

# 打印结果
print("预测结果（1表示正常点，-1表示异常点）：\n", results)

# 显示异常点的数量
print(f"异常点的数量：{list(results).count(-1)}")

"""
预测结果（1表示正常点，-1表示异常点）：
 [ 1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1 -1  1  1  1  1
  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
  1  1  1  1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1]
异常点的数量：11
"""