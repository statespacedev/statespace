import numpy as np
import matplotlib.pyplot as plt

def xts(xts, tts, end=None):
    x2 = len(tts)-1
    if not end == None:
        x2 = end
    plt.plot(tts[:x2], xts[:x2])
    plt.show()
    pass

def test(xhatts, xtilts, yhatts, ytillts, yts, Reets, tts):
    lw = 1
    plt.subplot(3,2,1)
    plt.plot(tts, xhatts, linewidth=lw)
    plt.subplot(3,2,2)
    plt.plot(tts, xtilts, linewidth=lw)
    plt.subplot(3,2,3)
    plt.plot(tts, yhatts, linewidth=lw)
    plt.subplot(3,2,4)
    plt.plot(tts, ytillts, linewidth=lw)
    plt.subplot(3,2,5)
    plt.plot(tts, yts, linewidth=lw)
    plt.subplot(3,2,6)
    plt.plot(tts, Reets, linewidth=lw)
    plt.show()
