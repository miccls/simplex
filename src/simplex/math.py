import numpy as np

def update_inverse(Binv, d, exiting_index):
    dl = d[exiting_index]
    rowl = Binv[exiting_index]
    Binv = np.array([row - (rowl * d[i]/dl) if i != exiting_index else row / dl for i, row in enumerate(Binv)])
    return Binv