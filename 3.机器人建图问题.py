import numpy as np
print("机器人建图问题")
precision = input("请输入精度(m)：")
Temp_yaw_str = input("请输入机器人当前方位角，单位为度(d)或弧度(r):")


def pos_calculate(yaw_0, precision):
    x_a, x_b = divmod(1.8, precision)
    y_a, y_b = divmod(1.5, precision)
    xy0 = np.array(range(1,))

