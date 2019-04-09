import numpy as np
import sys
from value_iteration import value_iteration
from policy_iteration import policy_iteration

if __name__ == "__main__":
    iterations = [value_iteration, policy_iteration]
    i = 0
    try:
        i = int(sys.argv[1])
    except IndexError:
        pass
    print('using', iterations[i].__name__)
    n = 3 * 4
    res = iterations[i](n)
    res.reverse()
    res = np.array(res)
    res.resize((3, 4))
    directions = dict([(0, '←'), (1, '→'), (2, '↓'), (3, '↑'), (None, 'o')])
    for row in res:
        for i in reversed(row):
            print(directions[i], end=' ')
        print()
