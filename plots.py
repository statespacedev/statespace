import numpy as np
import matplotlib.pyplot as plt

def xts(xts, tts, end=None):
    x2 = len(tts)-1
    if not end == None:
        x2 = end
    plt.plot(tts[:x2], xts[:x2])
    plt.show()
    pass