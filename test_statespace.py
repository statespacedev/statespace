import sys
sys.path.append('statespace')

def test_classical():
    from statespace import classical
    c = classical.Classical('extended', plot=False)
    assert(c.log[-1][1] > 2.08 and c.log[-1][1] < 2.12)

if __name__ == "__main__":
    test_classical()