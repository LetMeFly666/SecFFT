'''
Author: LetMeFly
Date: 2024-09-12 16:54:05
LastEditors: LetMeFly
LastEditTime: 2024-09-13 21:07:37
'''
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 生成示例数据
data = np.random.rand(10, 12)  # 10 行 12 列的随机数据

# 创建热力图
plt.figure(figsize=(10, 8))  # 设置图表大小
sns.heatmap(data, annot=True, cmap='YlGnBu')  # 使用绿色到蓝色的渐变配色
plt.title('Heatmap Example')  # 设置标题
plt.show()  # 显示热力图
plt.savefig('temp.relitu.jpg', format='jpg')