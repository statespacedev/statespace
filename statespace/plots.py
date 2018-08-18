import numpy as np
import matplotlib.pyplot as plt

def standard(tts, xhatts, xtilts, yhatts, ets, Reets):
    lw = 1
    plt.subplot(3,2,1)
    plt.plot(tts, xhatts, linewidth=lw)
    plt.subplot(3,2,2)
    plt.plot(tts, xtilts, linewidth=lw)
    plt.subplot(3,2,3)
    plt.plot(tts, yhatts, linewidth=lw)
    plt.subplot(3,2,4)
    plt.plot(tts, ets, linewidth=lw)
    plt.subplot(3,2,5)
    plt.plot(tts, Reets, linewidth=lw)
    plt.show()
