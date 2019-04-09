import numpy as np
from maze import get_neighbor, reward

def value_iteration(n, gamma=1, theta=1e-5):
    '''
    @description: 值迭代函数
    @param 
        n: 状态个数
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
