from sko.ACA import ACA_TSP
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
    return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])


# %% Do ACA

aca = ACA_TSP(func=cal_total_distance, n_dim=num_points,
              size_pop=50, max_iter=200,
              distance_matrix=distance_matrix)

best_x, best_y = aca.run()

# %% Plot
fig, ax = plt.subplots(1, 2)
best_points_ = np.concatenate([best_x, [best_x[0]]])
best_points_coordinate = points_coordinate[best_points_, :]
ax[0].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r')
pd.DataFrame(aca.y_best_history).cummin().plot(ax=ax[1])
plt.show()
