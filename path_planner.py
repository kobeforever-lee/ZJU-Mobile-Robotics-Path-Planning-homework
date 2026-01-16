import numpy as np
import heapq

class AStarPlanner:
    def __init__(self, env, resolution=0.5):
        self.env = env
        self.res = resolution  # 离散化步长
        # 定义 26 连通邻域（3D 空间中周围的所有方向）
        self.motions = [(dx, dy, dz) for dx in [-1, 0, 1] 
                                    for dy in [-1, 0, 1] 
                                    for dz in [-1, 0, 1] if not (dx==0 and dy==0 and dz==0)]

    def heuristic(self, p1, p2):
        """计算欧几里得距离作为启发式函数"""
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def plan(self, start, goal):
        start = tuple(start)
        goal = tuple(goal)

        # 优先队列：(优先级, 当前点)
        open_list = []
        heapq.heappush(open_list, (0 + self.heuristic(start, goal), start))
        
        # 记录路径和代价
        came_from = {start: None}
        g_score = {start: 0}

        while open_list:
            _, current = heapq.heappop(open_list)

            # 如果接近目标点，直接连接终点并返回
            if self.heuristic(current, goal) < self.res:
                return self.reconstruct_path(came_from, current, goal)

            for dx, dy, dz in self.motions:
                neighbor = (current[0] + dx * self.res, 
                            current[1] + dy * self.res, 
                            current[2] + dz * self.res)

                # 边界检查
                if self.env.is_outside(neighbor):
                    continue
                
                # 碰撞检查
                if self.env.is_collide(neighbor, epsilon=0.25): # 略微增加 epsilon 提高安全性
                    continue

                tentative_g_score = g_score[current] + self.heuristic(current, neighbor)

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(open_list, (f_score, neighbor))

        return None # 无法找到路径

    def reconstruct_path(self, came_from, current, goal):
        path = [goal]
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return np.array(path)

def find_path(env, start, goal):
    """
    供 main.py 调用的接口函数
    """
    planner = AStarPlanner(env, resolution=0.5)
    path = planner.plan(start, goal)
    return path
            











