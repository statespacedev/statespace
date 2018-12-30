import sys
sys.path.append('statespace')
import classical
import modern
import particle

tol1 = 1.

def test_classical_01():
    tmp = classical.Classical('ekf1', plot=False)
    assert(abs(tmp.log[-1][3]) < tol1)

def test_classical_02():
    tmp = classical.Classical('ekf2', plot=False)
    assert(abs(tmp.log[-1][3]) < tol1)

def test_modern_01():
    tmp = modern.Modern('ukf1', plot=False)
    assert(abs(tmp.log[-1][3]) < tol1)

def test_modern_02():
    tmp = modern.Modern('ukf2', plot=False)
    assert(abs(tmp.log[-1][3]) < tol1)

def test_particle_01():
    tmp = particle.Particle('pf1', plot=False)
    assert(abs(tmp.log[-1][3]) < tol1)

def test_particle_02():
    tmp = particle.Particle('pf2', plot=False)
    assert(abs(tmp.log[-1][3]) < tol1)
