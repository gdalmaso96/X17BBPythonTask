# Python implementation of momentum morphing class algorithm in https://arxiv.org/pdf/1410.7388.pdf
import numpy as np
from scipy.interpolate import RectBivariateSpline
# The class takes as input a numpy array with the first axis being the morphing parameter, the list of m parameters and the reference value of the morphing parameter

class MorphTemplate1D:
    def __init__(self, hist, ms, m0):
        self.hist = hist
        self.originalMS = ms
        self.maxMs = max(ms)
        self.minMs = min(ms)
        self.n = len(self.hist)
        self.ms = ms
        self.m0 = m0
        self.m0Index = self.findm0()
        self.ms = self.ChebyshevNodes(ms)
        self.M = self.computMmatrix()
    
    # Compute Chebyshev nodes
    def ChebyshevNodes(self, ms):
        k = (ms - self.minMs)/(self.maxMs - self.minMs)*(self.n-1) + 1
        res = np.cos((k - 1)*np.pi/(self.n-1))
        return res
    
    # Find the index of m0 in ms
    def findm0(self):
        if self.m0 not in self.ms:
            return 0
        else:
            return np.where(self.ms == self.m0)[0][0]
    
    # Compute the M matrix
    def computMmatrix(self):
        n = len(self.hist)
        M = []
        for i in range(n): # loop on morphs
            m = []
            for j in range(n): # loop on powers
                m.append(np.power((self.ms[i] - self.ms[self.m0Index]), j))
            M.append(np.array(m))
        M = np.array(M)
        M = np.linalg.inv(M).transpose()
        return M
    
    # Compute ci coefficients
    def computCi(self, m):
        deltam = np.power(m - self.ms[self.m0Index], np.linspace(0, len(self.hist) - 1, len(self.hist)))
        c = np.dot(self.M, deltam)
        return c

    # Compute the morphed template
    def morphTemplate(self, x):
        # Apply Chebyshev nodes
        smear = self.ChebyshevNodes(x)
        ci = self.computCi(smear)
        thist = np.swapaxes(self.hist, 0, len(self.hist.shape) - 1)
        res = np.dot(thist, ci).transpose()
        res = res*(res > 0)
        return res

class MorphTemplate2D:
    # Here m0 is a point in 2D
    # ms is a list of points in 2D
    # the first point is the reference point
    def __init__(self, hist, ms, m0Index):
        self.hist = hist
        self.ms = ms
        self.m0Index = m0Index
        self.n1 = self.hist.shape[0]
        self.n2 = self.hist.shape[1]
        self.flattenHist()
        self.M = self.computMmatrix()

    # Flatten the histogram
    def flattenHist(self):
        newHist = []
        for i in range(self.n1):
            for j in range(self.n2):
                newHist.append(self.hist[i][j])
        self.hist = np.array(newHist)
    
    # Compute the M matrix
    def computMmatrix(self):
        M = []
        for i in range(self.n1*self.n2): # loop on morphs
            m = []
            for j in range(self.n1): # loop on powers first index
                for k in range(self.n2): # loop on powers second index
                    m.append(np.power((self.ms[i][0] - self.ms[self.m0Index][0]), j)*np.power((self.ms[i][1] - self.ms[self.m0Index][1]), k))
            M.append(np.array(m))
        M = np.array(M)
        M = np.linalg.inv(M).transpose()
        
        return M
    
    # Compute ci coefficients
    def computeCi(self, m):
        deltam = []
        for i in range(self.n1):
            for j in range(self.n2):
                deltam.append(np.power((m[0] - self.ms[self.m0Index][0]), i)*np.power((m[1] - self.ms[self.m0Index][1]), j))
        deltam = np.array(deltam)
        c = np.dot(self.M, deltam)
        c = np.abs(c*(c > 1e-8))
        return c

    # Compute the morphed template
    def morphTemplate(self, smear):
        c = self.computeCi(smear)
        thist = np.swapaxes(self.hist, 0, len(self.hist.shape) - 1)
        
        res = np.dot(thist, c).transpose()
        res = res*(res > 0)
        return res

    # Compute the morphed template
    def morphTemplateVar(self, smear):
        c = np.power(self.computeCi(smear), 2)
        thist = np.swapaxes(self.hist, 0, len(self.hist.shape) - 1)
        
        res = np.dot(thist, c).transpose()
        res = res*(res > 0)
        return res
