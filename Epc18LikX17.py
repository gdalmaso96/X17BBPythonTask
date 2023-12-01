# X17 PDF
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
import sample2D


# PDF from Francesco
class AngleVSEnergySum():
    def __init__(self, minAngle, maxAngle, nAngleBins, minEsum, maxEsum, nEsumBins):
        self.minAngle = minAngle
        self.maxAngle = maxAngle
        self.nAngleBins = nAngleBins
        self.minEsum = minEsum
        self.maxEsum = maxEsum
        self.nEsumBins = nEsumBins
        self.normalization = quad(self.pdfAngle, minAngle, maxAngle)[0]*quad(self.pdfEsum, minEsum, maxEsum)[0]/((maxAngle - minAngle)/nAngleBins*(maxEsum - minEsum)/nEsumBins)
    
    def pdfEsum(self, esum):
        y = sample2D.eEPC18(esum, *sample2D.esumEpc18)
        return y*(y > 0)
    
    def binnedPdfEsum(self, esum):
        binContent = np.array([])
        for e in esum:
            deltaEsum = e - self.minEsum
            deltaEsum /= (self.maxEsum - self.minEsum)
            deltaEsum *= self.nEsumBins
            deltaEsum = int(deltaEsum)
            binLowEdge = self.minEsum + deltaEsum*(self.maxEsum - self.minEsum)/self.nEsumBins
            binHighEdge = self.minEsum + (deltaEsum+1)*(self.maxEsum - self.minEsum)/self.nEsumBins
            binContent = np.append(binContent, quad(self.pdfEsum, binLowEdge, binHighEdge)[0]/(self.maxEsum - self.minEsum)*self.nEsumBins)
        return binContent

    def pdfAngle(self, dth):
        y = sample2D.tEPC18(dth, *sample2D.rangEpc18)
        return y*(y > 0)
    
    def binnedPdfAngle(self, dth):
        binContent = np.array([])
        for d in dth:
            deltaDth = d - self.minAngle
            deltaDth /= (self.maxAngle - self.minAngle)
            deltaDth *= self.nAngleBins
            deltaDth = int(deltaDth)
            binLowEdge = self.minAngle + deltaDth*(self.maxAngle - self.minAngle)/self.nAngleBins
            binHighEdge = self.minAngle + (deltaDth+1)*(self.maxAngle - self.minAngle)/self.nAngleBins
            binContent = np.append(binContent, quad(self.pdfAngle, binLowEdge, binHighEdge)[0]/(self.maxAngle - self.minAngle)*self.nAngleBins)
        return binContent
        
    
    def pdf(self, dth, esum):
        dth = np.array(dth)
        esum = np.array(esum)
        results = np.array([])
        for x, y in zip(dth, esum):
            pEsum = self.binnedPdfEsum(y)
            pRelAngle = self.binnedPdfAngle(x)
            #pEsum = self.pdfEsum(y)
            #pRelAngle = self.pdfAngle(x)
            results = np.append(results, pRelAngle*pEsum/self.normalization)
        
        results = results.reshape(self.nEsumBins, self.nAngleBins)
        results = results.transpose()
    
        return results