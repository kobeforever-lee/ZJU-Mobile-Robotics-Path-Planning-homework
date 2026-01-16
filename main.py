from flight_environment import FlightEnvironment
from path_planner import find_path
from trajectory_generator import TrajectoryGenerator
import numpy as np

# 初始化环境 (随机生成50个障碍物)
env = FlightEnvironment(50)
start = (1, 2, 0)
goal = (18, 18, 3)

# --------------------------------------------------------------------------------------------------- #
# 第一步: 路径规划 (Path Planning) - Task A
# --------------------------------------------------------------------------------------------------- #
print("正在规划路径...")
# 调用 A* 算法寻找路径
path = find_path(env, start, goal)

if path is not None:
    print("路径规划成功！")
    # 绘制 3D 路径图 (第一张结果图)
    # 注意：env.plot_cylinders 会阻塞程序，关闭弹出的窗口后程序才会继续运行
    env.plot_cylinders(path) 
else:
    print("未能找到路径！")
    exit() # 如果没找到路径，终止程序

# --------------------------------------------------------------------------------------------------- #
# 第二步: 轨迹规划 (Trajectory Planning) - Task B
# --------------------------------------------------------------------------------------------------- #
print("正在生成平滑轨迹...")

# 初始化轨迹生成器，传入上一步生成的离散路径
# avg_speed 设置为 2.5 m/s
traj_gen = TrajectoryGenerator(path, avg_speed=2.5)

# 计算三次样条插值
traj_gen.solve()

# 绘制 x-t, y-t, z-t 曲线图 (第二张结果图)
traj_gen.plot_trajectory()

# --------------------------------------------------------------------------------------------------- #

