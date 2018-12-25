import sys
sys.path.append('statespace')

lo, hi = 2.07, 2.13

from statespace import classical

def test_classical_001():
    tmp = classical.Classical('linearized', plot=False)
    assert(tmp.log[-1][1] > lo and tmp.log[-1][1] < hi)

def test_classical_002():
    tmp = classical.Classical('extended', plot=False)
    assert(tmp.log[-1][1] > lo and tmp.log[-1][1] < hi)

def test_classical_003():
    tmp = classical.Classical('adaptive', plot=False)
    assert(tmp.log[-1][1] > lo and tmp.log[-1][1] < hi)


from statespace import modern

def test_modern_001():
    tmp = modern.Modern('sigmapoint', plot=False)
    assert(tmp.log[-1][1] > lo and tmp.log[-1][1] < hi)
