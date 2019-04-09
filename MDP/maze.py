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
