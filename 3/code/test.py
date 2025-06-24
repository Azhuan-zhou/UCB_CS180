import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon

# 创建一个空白图像
image = np.zeros((100, 100), dtype=np.uint8)

# 定义多边形的顶点
r = np.array([20, 50, 80, 50])
c = np.array([20, 10, 80, 90])

# 获取多边形的坐标
rr, cc = polygon(r, c)

# 在图像中绘制多边形
image[rr, cc] = 1

# 显示图像
plt.imshow(image, cmap='gray')
plt.show()