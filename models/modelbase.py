
class ModelBase:
    def __init__(self):
        self.tsteps = 151
        self.dt = .01
        self.x = 2.
        self.Rww = 1e-6
        self.Rvv = 9e-2
        self.log = []
        self.custom()

    def steps(self):
        pass


