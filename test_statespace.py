import sys
sys.path.append('statespace')

tol1 = 1.

from statespace import classical

def test_classical_001():
    tmp = classical.Classical('kf', plot=False)
    assert(abs(tmp.log[-1][3]) < tol1)

def test_classical_002():
    tmp = classical.Classical('ekf1', plot=False)
    assert(abs(tmp.log[-1][3]) < tol1)

def test_classical_003():
    tmp = classical.Classical('ekf2', plot=False)
    assert(abs(tmp.log[-1][3]) < tol1)

from statespace import modern

def test_modern_001():
    tmp = modern.Modern('ukf1', plot=False)
    assert(abs(tmp.log[-1][3]) < tol1)

def test_modern_002():
    tmp = modern.Modern('ukf2', plot=False)
    assert(abs(tmp.log[-1][3]) < tol1)

from statespace import particle

def test_particle_001():
    tmp = particle.Particle('pf1', plot=False)
    assert(abs(tmp.log[-1][3]) < tol1)
