# X17 PDF
import numpy as np
from scipy.stats import norm

electronMass = 0.51099895000
transitionEnergy = 18.15

def maxRelAngle(mass, energy):
    gamma = energy/mass
    betag = np.sqrt(gamma**2 - 1)
    
    # Maximum dth is given by transforming momenta when pxcm = 0
    EposCM = mass/2
    pyCM = np.sqrt(EposCM**2 - electronMass**2)
    pxLab = betag*EposCM
    
    tanA = pyCM/pxLab
    return 2*np.arctan(tanA)*180/np.pi

# PDF from Francesco
def AngleVSEnergySum(dth, esum, mass, dthMin, dthMax, dthnBins, esumMin, esumMax, esumnBins, dthRes = 9.5, esumRes = 1.15):
    dth = np.array(dth)
    esum = np.array(esum)
    
    dThCenter = maxRelAngle(mass, transitionEnergy)
    
    results = np.array([])
    for x, y in zip(dth, esum):
        pEsum = norm.pdf(y, loc = transitionEnergy, scale = esumRes)*(esumMax - esumMin)/esumnBins #/(norm.cdf(esumMax, loc = transitionEnergy, scale = esumRes) - norm.cdf(esumMin, loc = transitionEnergy, scale = esumRes))
        pRelAngle = norm.pdf(x, loc = dThCenter, scale = dthRes)*(dthMax - dthMin)/dthnBins #/(norm.cdf(dthMax, loc = dThCenter, scale = dthRes) - norm.cdf(dthMin, loc = dThCenter, scale = dthRes))
        results = np.append(results, pRelAngle*pEsum)
    
    results = results.reshape(esumnBins, dthnBins)
    results = results.transpose()
    
    return results
    
    
# Random PDF
def MassVSEnergySum(imas, esum, mass, imasMin, imasMax, imasnBins, esumMin, esumMax, esumnBins, imasRes = 9.5, esumRes = 1.15):
    imas = np.array(imas)
    esum = np.array(esum)
    
    results = np.array([])
    for x, y in zip(imas, esum):
        pEsum = norm.pdf(y, loc = transitionEnergy, scale = esumRes)*(esumMax - esumMin)/esumnBins #/(norm.cdf(esumMax, loc = transitionEnergy, scale = esumRes) - norm.cdf(esumMin, loc = transitionEnergy, scale = esumRes))
        pInvMass = norm.pdf(x, loc = mass, scale = imasRes)*(imasMax - imasMin)/imasnBins #/(norm.cdf(dthMax, loc = dThCenter, scale = dthRes) - norm.cdf(dthMin, loc = dThCenter, scale = dthRes))
        results = np.append(results, pInvMass*pEsum)
    
    results = results.reshape(esumnBins, imasnBins)
    results = results.transpose()
    
    return results
    
    
