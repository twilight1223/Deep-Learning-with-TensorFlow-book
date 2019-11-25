# __author__:zsshi
# __date__:2019/11/22

#matplotlibrc配置文件

import matplotlib
print(matplotlib.rcParams)

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2 * np.pi)
y = np.sin(x)

matplotlib.rcParams['lines.color'] = 'blue'  # 更改划线颜色的默认设置

plt.plot(x, y, label='sin', linewidth=5)
plt.legend()
plt.show()
