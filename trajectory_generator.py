import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

class TrajectoryGenerator:
    def __init__(self, path, avg_speed=2.0):
        """
        初始化轨迹生成器
        :param path: Nx3 的 numpy array，包含路径点 (x, y, z)
        :param avg_speed: 机器人的平均移动速度，用于计算时间
        """
        self.path = path
        self.avg_speed = avg_speed
        self.time_knots = None
        self.cs_x = None
        self.cs_y = None
        self.cs_z = None
        self.total_time = 0

    def solve(self):
        """
        生成平滑轨迹的核心函数
        使用 Scipy 的 CubicSpline (三次样条插值) 进行平滑处理
        """
        # 1. 计算每个路径点之间的欧几里得距离
        diffs = np.diff(self.path, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        
        # 2. 根据距离和速度计算累计时间 (Time allocation)
        # 第一个点时间为 0
        times = [0]
        current_time = 0
        for d in dists:
            # 简单假设匀速运动，时间 = 距离 / 速度
            dt = d / self.avg_speed
            # 避免两个点重合导致 dt 为 0，设置一个极小值
            if dt < 1e-6:
                dt = 0.1
            current_time += dt
            times.append(current_time)
            
        self.time_knots = np.array(times)
        self.total_time = current_time

        # 3. 分别对 x, y, z 进行三次样条插值 (Cubic Spline Interpolation)
        # bc_type='natural' 表示边界处二阶导数为0，使曲线两端较平缓
        self.cs_x = CubicSpline(self.time_knots, self.path[:, 0], bc_type='natural')
        self.cs_y = CubicSpline(self.time_knots, self.path[:, 1], bc_type='natural')
        self.cs_z = CubicSpline(self.time_knots, self.path[:, 2], bc_type='natural')

        print(f"Trajectory generated. Total time: {self.total_time:.2f} s")

    def get_trajectory_points(self, num_points=200):
        """
        获取密集的时间点和对应的坐标，用于绘图或控制
        """
        t_dense = np.linspace(0, self.total_time, num_points)
        x_dense = self.cs_x(t_dense)
        y_dense = self.cs_y(t_dense)
        z_dense = self.cs_z(t_dense)
        return t_dense, x_dense, y_dense, z_dense

    def plot_trajectory(self):
        """
        绘制题目要求的 x-t, y-t, z-t 波形图，并标记原始路径点
        (改为英文标签以解决乱码问题)
        """
        if self.time_knots is None:
            print("Please call solve() first.")
            return

        # 获取用于画图的密集点
        t_dense, x_dense, y_dense, z_dense = self.get_trajectory_points(500)
        
        # 原始离散路径点
        x_knots = self.path[:, 0]
        y_knots = self.path[:, 1]
        z_knots = self.path[:, 2]

        # 建立 3 个子图
        fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
        
        # X-t 图
        axes[0].plot(t_dense, x_dense, 'r-', label='Trajectory X')
        axes[0].plot(self.time_knots, x_knots, 'bo', label='Path Points')
        axes[0].set_ylabel('X [m]')
        axes[0].set_title('Trajectory Coordinates vs Time')
        axes[0].legend()
        axes[0].grid(True)

        # Y-t 图
        axes[1].plot(t_dense, y_dense, 'g-', label='Trajectory Y')
        axes[1].plot(self.time_knots, y_knots, 'bo', label='Path Points')
        axes[1].set_ylabel('Y [m]')
        axes[1].legend()
        axes[1].grid(True)

        # Z-t 图
        axes[2].plot(t_dense, z_dense, 'b-', label='Trajectory Z')
        axes[2].plot(self.time_knots, z_knots, 'bo', label='Path Points')
        axes[2].set_ylabel('Z [m]')
        axes[2].set_xlabel('Time [s]')
        axes[2].legend()
        axes[2].grid(True)

        plt.tight_layout()
        plt.show()