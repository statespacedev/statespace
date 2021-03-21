
class ModelBase:

    def ekf(self): return None
    def ekfud(self): return None
    def sp(self): return None
    def pf(self): return None

    def __init__(self):
        self.log = []

    def steps(self): pass

    def f(self, x): return None

    def h(self, x): return None

    def F(self, x): return None

    def H(self, x): return None

class SPKFBase():

    def __init__(self):
        self.vf = None
        self.vh = None

    def X1(self, xhat, Ptil): return None

    def X2(self, X): return None

class PFBase():

    def __init__(self): pass

    def F(self, x): return None

    def H(self, y, x): return None

if __name__ == "__main__":
    pass