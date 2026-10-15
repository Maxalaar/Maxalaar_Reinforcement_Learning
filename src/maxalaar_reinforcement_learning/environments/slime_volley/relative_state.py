import numpy as np


class RelativeState:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.vx = 0
        self.vy = 0
        self.bx = 0
        self.by = 0
        self.bvx = 0
        self.bvy = 0
        self.ox = 0
        self.oy = 0
        self.ovx = 0
        self.ovy = 0

    def getObservation(self):
        result = [self.x, self.y, self.vx, self.vy,
                  self.bx, self.by, self.bvx, self.bvy,
                  self.ox, self.oy, self.ovx, self.ovy]
        scaleFactor = 10.0
        result = np.array(result) / scaleFactor

        # result = np.array([1.2000, 0.1500, 0.0000, 0.0000, 0.0000, 1.2000, -0.8870, 1.8969, 1.2000, 0.1500, 0.0000, 0.0000])
        # result = np.array([ 2.25 ,  0.446,  0.   , -0.414, -0.739,  1.884, -1.109,  0.096,  1.2  ,  0.15 ,  0.   ,  0.   ])
        return result