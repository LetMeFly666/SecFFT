'''
Author: LetMeFly
Date: 2024-09-04 10:33:08
LastEditors: LetMeFly
LastEditTime: 2024-09-04 10:33:19
'''
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# 示例数据，假设有10个圆（实际数据需要替换）
data = {
    'x': [2, 3, 5, 8, 13, 21, 34, 55, 89, 144],  # 圆心x坐标
    'y': [3, 5, 7, 11, 17, 27, 37, 47, 60, 75],  # 圆心y坐标
    'radius': [1, 1.2, 0.8, 1.5, 2.0, 1.7, 1.1, 3.0, 4.5, 5.0]  # 圆的半径，代表置信度
}

# 转化为DataFrame
df = pd.DataFrame(data)

# 使用Isolation Forest进行异常检测
model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)  # contamination为预期异常点比例
df['scores'] = model.fit_predict(df[['x', 'y', 'radius']])
df['anomaly_score'] = model.decision_function(df[['x', 'y', 'radius']])

# 可视化异常检测结果
plt.figure(figsize=(10, 6))
plt.scatter(df['x'], df['y'], c=df['scores'], cmap='coolwarm', edgecolor='k', s=100)
plt.xlabel('x - 圆心x坐标')
plt.ylabel('y - 圆心y坐标')
plt.title('基于Isolation Forest的异常检测')
plt.colorbar(label='异常得分')
plt.show()

# 显示异常点
anomalies = df[df['scores'] == -1]
print("异常点:")
print(anomalies)
