import numpy as np
from functools import reduce

def reward(s):
    '''
    @description: 价值函数
    @param 
        s: 状态，[0, 11]
    @return: s的效用值（两个终点分别为1和-1，其他为-0.03）
    '''
    if s == 11:
        return 1.0
    if s == 7:
        return -1.0
    return -0.03

def get_neighbor(s):
    '''
    @description: 一个状态向前以及垂直方向的三个邻居
    @param 
        s: 状态
    @return: 三个邻居及其概率
    '''
    def wall_wrap(i, s):
        if i == 5:
            return s
        r, c = s // 4, s % 4
        if r == 0 and i == s - 4 or r == 2 and i == s + 4:
            return s
        if c == 0 and i == s - 1 or c == 3 and i == s + 1:
            return s
        return i
        
    neighbors = dict()
    if s == 11 or s == 7 or s == 5:     # 终点和墙不能作为起始状态，无邻居
        return neighbors
    for d in (-1, 1, -4, 4):    # 四个方向
        neighbors.setdefault(d, [])
        forward = wall_wrap(s + d, s)
        neighbors[d].append((0.8, forward)) # 向前的概率是0.8

        v = 4 if abs(d) == 1 else 1
        vertical = [wall_wrap(s - v, s), wall_wrap(s + v, s)]
        for v in vertical:
            neighbors[d].append((0.1, v))
    return neighbors

def value_iteration(n, R, gamma=1, theta=1e-5):
    '''
    @description: 值迭代函数
    @param 
        n: 状态个数
        R: 效用值函数
        gamma: 折扣
        theta: 收敛误差
    @return: 最优行动策略
    '''
    optimal_V = [None for i in range(n)]    # 全局最优的V
    bests = [None for i in range(n)]    # 最优行动

    neighbors = [get_neighbor(i) for i in range(n)]
    V = np.zeros(n)
    while True:
        new_V = np.zeros(n)
        for s in range(n):
            neighbor = neighbors[s]
            max_utility = 0
            best_action = None
            if neighbor:
                # 对每种行动，遍历可能的邻居，对它们求和。最后得出和最大的行动及其效用值
                best_action, max_utility = max([(action, sum(map(lambda st: st[0] * V[st[1]], states))) 
                                         for action, states in neighbor.items()], 
                                        key=lambda x: x[1])
            new_V[s] = max_utility * gamma + reward(s)  # 用AI书上的贝尔曼公式
            if optimal_V[s] is None or new_V[s] > optimal_V[s]: # 新的全局最优
                optimal_V[s] = new_V[s]
                bests[s] = best_action
        if np.max(np.abs(V - new_V)) < theta:   # 收敛
            break
        V = new_V
    return bests

if __name__ == "__main__":
    n = 3 * 4
    print(value_iteration(n, reward))
