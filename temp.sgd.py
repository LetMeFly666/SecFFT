'''
Author: LetMeFly
Date: 2024-09-06 15:38:44
LastEditors: LetMeFly
LastEditTime: 2024-09-06 15:38:52
'''
import matplotlib.pyplot as plt
import numpy as np

# 创建数据
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2  # 损失函数的例子，这里使用了一个简单的二次函数

# 绘制等高线图
plt.figure(figsize=(6, 6))
plt.contour(X, Y, Z, levels=20, cmap='RdGy')

# 绘制梯度下降路径
x_path = [-2, -1.5, -1, -0.7, -0.5, -0.3, -0.1, 0]
y_path = [-2, -1.2, -0.8, -0.5, -0.3, -0.1, -0.05, 0]
plt.plot(x_path, y_path, marker='o', color='blue', linestyle='-', linewidth=1, markersize=5)

# 标注起点和终点
plt.text(x_path[0], y_path[0], 'Start', fontsize=12, ha='right')
plt.text(x_path[-1], y_path[-1], 'End', fontsize=12, ha='left')

# 设置图形标题和坐标轴
plt.title('Gradient Descent Path on Contour Plot')
plt.xlabel('X')
plt.ylabel('Y')

# 保存为SVG文件
plt.savefig('gradient_descent.svg', format='svg')

# 显示图形
plt.show()
