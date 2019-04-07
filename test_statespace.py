import sys
sys.path.append('statespace')
import kalman, sigmapoint, particle

tol1 = 1.

def test_classical_00():
    tmp = kalman.Classical('rccircuit', plot=False)
    assert(abs(tmp.innov.log[-1][3]) < tol1)

def test_classical_01():
    tmp = kalman.Classical('kf2', plot=False)
    assert(abs(tmp.innov.log[-1][3]) < tol1)

def test_classical_02():
    tmp = kalman.Classical('ekf1', plot=False)
    assert(abs(tmp.innov.log[-1][3]) < tol1)

def test_classical_03():
    tmp = kalman.Classical('ekf2', plot=False)
    assert(abs(tmp.innov.log[-1][3]) < tol1)

def test_modern_01():
    tmp = sigmapoint.Modern('spkf1', plot=False)
    assert(abs(tmp.innov.log[-1][3]) < tol1)

def test_modern_02():
    tmp = sigmapoint.Modern('spkf2', plot=False)
    assert(abs(tmp.innov.log[-1][3]) < tol1)

def test_particle_01():
    tmp = particle.Particle('pf1')
    assert(abs(tmp.innovs.log[-1][3]) < tol1)

def test_particle_02():
    tmp = particle.Particle('pf2')
    assert(abs(tmp.innovs.log[-1][3]) < tol1)
