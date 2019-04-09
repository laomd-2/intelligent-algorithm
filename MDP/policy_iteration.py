import numpy as np
from maze import reward, get_neighbor

def polciy_evaluation(n, policy, gamma=1.0, theta=1e-5):
    '''
    @description: 根据当前策略计算V数组
    @param 
        n: 状态数
        policy: 当前的策略 
    @return: V数组
    '''
    neighbors = [get_neighbor(i) for i in range(n)]
    V = np.zeros(n)
    while True:
        new_V = np.zeros(n)
        for s in range(n):
            neighbor = neighbors[s]
            value = 0
            if neighbor:
                # 计算当前策略对应的行动下的value值
                value = sum(map(lambda st: st[0] * V[st[1]], neighbor[policy[s]]))
            new_V[s] = value * gamma + reward(s)  # 用AI书上的贝尔曼公式
        if np.max(np.abs(V - new_V)) < theta:   # 收敛
            break
        V = new_V
    return V

def policy_iteration(n, gamma=1.0, theta=1e-5):
    policy = [1] * n  # 初始策略为全部向右走
    policy[5] = policy[7] = policy[11] = None

    stable = False
    neighbors = [get_neighbor(i) for i in range(n)]
    while not stable:
        stable = True
        # 策略迭代，计算出当前的V
        V = polciy_evaluation(n, policy, gamma, theta)
        # policy improvement
        for s in range(n):
            neighbor = neighbors[s]
            if neighbor:
                temp = policy[s]
                # 实际的最佳策略
                policy[s] = max([(action, sum(map(lambda st: st[0] * V[st[1]], states))) 
                                                for action, states in neighbor.items()], 
                                                key=lambda x: x[1])[0]
                if temp != policy[s]:   # 策略未收敛
                    stable = False
    return policy
