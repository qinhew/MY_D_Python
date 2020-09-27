import numpy as np
from scipy import spatial
import pandas as pd
import matplotlib.pyplot as plt

num_points = 25

points_coordinate = np.random.rand(
    num_points, 2)  # generate coordinate of points
# 计算两个输入集合的距离，通过metric参数指定计算距离的不同方式得到不同的距离度量值
distance_matrix = spatial.distance.cdist(
    points_coordinate, points_coordinate, metric='euclidean')


def cal_total_distance(routine):
    num_points, = routine.shape
    temp_2 = [distance_matrix[routine[k2 % num_points],
                              routine[(k2 + 1) % num_points]]for k2 in range(num_points)]
    return sum(temp_2)


n_dim = num_points  # 城市数量
size_pop = 50  # 蚂蚁数量
max_iter = 200  # 迭代次数
alpha = 1  # 信息素重要程度
beta = 2  # 适应度的重要程度
rho = 0.1  # 信息素挥发速度

prob_matrix_distance = 1 / \
    (distance_matrix + 1e-10 * np.eye(n_dim, n_dim))  # 避免除零错误，eye生成对角矩阵
Tau = np.ones((n_dim, n_dim))  # 信息素矩阵，每次迭代都会更新，元素下标代表路线的信息素
Table = np.zeros((size_pop, n_dim)).astype(np.int)  # 某一代每个蚂蚁的爬行路径
y = None  # 某一代每个蚂蚁的爬行总距离
generation_best_X, generation_best_Y = [], []  # 记录各代的最佳情况
x_best_history, y_best_history = generation_best_X, generation_best_Y  # 历史原因，为了保持统一
best_x, best_y = None, None

for i in range(max_iter):  # 对每次迭代
    prob_matrix = (Tau ** alpha) * \
        (prob_matrix_distance) ** beta  # 转移概率，无须归一化。
    for j in range(size_pop):  # 对每个蚂蚁
        Table[j, 0] = 0  # start point，其实可以随机，但没什么区别
        for k in range(n_dim - 1):  # 蚂蚁到达的每个节点
            taboo_set = set(Table[j, :k + 1])  # 已经经过的点和当前点，不能再次经过
            allow_list = list(
                set(range(n_dim)) - taboo_set)  # 在这些点中做选择
            prob = prob_matrix[Table[j, k], allow_list]
            prob = prob / prob.sum()  # 概率归一化
            next_point = np.random.choice(
                allow_list, size=1, p=prob)[0]
            Table[j, k + 1] = next_point

    # 计算距离
    temp_1 = [cal_total_distance(i1) for i1 in Table]
    y = np.array(temp_1)

    # 顺便记录历史最好情况
    index_best = y.argmin()
    x_best, y_best = Table[index_best,
                           :].copy(), y[index_best].copy()
    generation_best_X.append(x_best)
    generation_best_Y.append(y_best)

    # 计算需要新涂抹的信息素
    delta_tau = np.zeros((n_dim, n_dim))
    for j1 in range(size_pop):  # 每个蚂蚁
        for k1 in range(n_dim - 1):  # 每个节点
            # 蚂蚁从n1节点爬到n2节点
            n1, n2 = Table[j1, k1], Table[j1, k1 + 1]
            delta_tau[n1, n2] += 1 / y[j1]  # 涂抹的信息素
        n1, n2 = Table[j1, n_dim - 1], Table[j1, 0]  # 蚂蚁从最后一个节点爬回到第一个节点
        delta_tau[n1, n2] += 1 / y[j1]  # 涂抹信息素

    # 信息素飘散+信息素涂抹
    Tau = (1 - rho) * Tau + delta_tau

best_generation = np.array(generation_best_Y).argmin()
best_x = generation_best_X[best_generation]
best_y = generation_best_Y[best_generation]

fig, ax = plt.subplots(1, 2)
best_points_ = np.concatenate([best_x, [best_x[0]]])
best_points_coordinate = points_coordinate[best_points_, :]
ax[0].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r')
pd.DataFrame(y_best_history).cummin().plot(ax=ax[1])
plt.show()
