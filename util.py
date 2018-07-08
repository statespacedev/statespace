import numpy as np

def div0(a, b):
    try:
        return a / float(b)
    except:
        return np.nan