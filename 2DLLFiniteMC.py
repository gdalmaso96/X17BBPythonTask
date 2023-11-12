# This routine fits a 2D histogram with the Beeston-Barlow Likelihood
# The data and MC are a complete X17 production

import uproot
import numpy as np
from scipy.integrate import quad
from matplotlib import pyplot as plt
from scipy.optimize import brentq
from iminuit import Minuit
import matplotlib
from scipy.stats import chi2, norm
import time

matplotlib.rcParams.update({'font.size': 30})

dataFile = 'X17MC2021.root'
MCFile = 'X17reference.root'

dthMin = 20
dthMax = 180
dthnBins = 20

esumMin = 10
esumMax = 24
esumnBins = 14

imasMin = 12
imasMax = 20
imasnBins = 40

VariableSelection = 0 # 0: dth, 1: imas

startingPs = np.array([450, 37500, 27500, 135000, 50000])

# Import MC histogram
def loadMC(fileName):
    with uproot.open(fileName + ':ntuple') as MC:
        x = MC.arrays(['imas', 'ecode', 'esum', 'dth'], library='np')
        if VariableSelection == 0:
            var = 'dth'
        elif VariableSelection == 1:
            var = 'imas'
            
        vXMC   = x[var][x['ecode'] == 0]
        vE15MC = x[var][x['ecode'] == 1]
        vI15MC = x[var][x['ecode'] == 2]
        vE18MC = x[var][x['ecode'] == 3]
        vI18MC = x[var][x['ecode'] == 4]
        
        eXMC   = x['esum'][x['ecode'] == 0]
        eE15MC = x['esum'][x['ecode'] == 1]
        eI15MC = x['esum'][x['ecode'] == 2]
        eE18MC = x['esum'][x['ecode'] == 3]
        eI18MC = x['esum'][x['ecode'] == 4]
        
        # Create histograms
        hXMC, binsXMCx, binsXMCy     = np.histogram2d(vXMC, eXMC, bins=[dthnBins, esumnBins], range=[[dthMin, dthMax], [esumMin, esumMax]])
        hE15MC, binsE15MCx, binsE15MCy = np.histogram2d(vE15MC, eE15MC, bins=[dthnBins, esumnBins], range=[[dthMin, dthMax], [esumMin, esumMax]])
        hI15MC, binsI15MCx, binsI15MCy = np.histogram2d(vI15MC, eI15MC, bins=[dthnBins, esumnBins], range=[[dthMin, dthMax], [esumMin, esumMax]])
        hE18MC, binsE18MCx, binsE18MCy = np.histogram2d(vE18MC, eE18MC, bins=[dthnBins, esumnBins], range=[[dthMin, dthMax], [esumMin, esumMax]])
        hI18MC, binsI18MCx, binsI18MCy = np.histogram2d(vI18MC, eI18MC, bins=[dthnBins, esumnBins], range=[[dthMin, dthMax], [esumMin, esumMax]])
    
    hMX = np.array([hXMC, hE15MC, hI15MC, hE18MC, hI18MC])
    return hMX, binsXMCx, binsXMCy

# Import data histogram
def loadData(fileName):
    with uproot.open(fileName + ':ntuple') as data:
        x = data.arrays(['imas', 'ecode', 'esum', 'dth'], library='np')
        if VariableSelection == 0:
            var = 'dth'
        elif VariableSelection == 1:
            var = 'imas'
            
        vXdata   = x[var]
        eXdata   = x['esum']
        
        # Create histograms
        hdata, binsdatax, binsdatay = np.histogram2d(vXdata, eXdata, bins=[dthnBins, esumnBins], range=[[dthMin, dthMax], [esumMin, esumMax]])
    
    return hdata, binsdatax, binsdatay

# Define Aji
# -- aji: MC ecode j, population of bin i
# -- pj: population j weight
# -- di: data population of bin i
# -- fi: MC prediction of population of bin i
def Aji(aji, pj, ti):
    return aji/(1 + ti*pj)

# Define ti solver
# Ignore the null case for the moment
# -- di: data population of bin i
# -- p: array of pj
# -- ai: array of aji
def solveTi(ti, di, p, ai):
    A = di/(1 - ti)
    B = p*ai/(1 + ti*p)
    B = B.sum()
    return A - B

def LogLikelihood(p0, p1, p2, p3, p4, hdata, hMX, getA=False):
    LL = 0
    AIJ = []
    p = np.array([p0, p1, p2, p3, p4])
    K = np.argsort(p)[::-1]
    
    # Run through bins
    if VariableSelection == 0:
        nBins = dthnBins
    elif VariableSelection == 1:
        nBins = imasnBins
    for I in range(nBins):
        for J in range(esumnBins):
            Di = hdata[I][J]
            
            ai = []
            Ai = []
            for i in range(len(hMX)):
                ai.append(hMX[i][I][J])
                Ai.append(hMX[i][I][J])
            ai = np.array(ai)
            Ai = np.array(Ai)
            
            # Do null bins a la Beeston-Barlow
            doNormalBB = True
            if (ai[K[0]] == 0):
                Ai[K[0]] = Di/(1 + p[K[0]])
            
                for k in K[1:]:
                    Ai[K[0]] += -p[k]*ai[k]/(p[K[0]] - p[k])
                    Ai[k] = ai[k]/(1 - p[k]/p[K[0]])
                if Ai[K[0]] <= 0:
                    Ai[K[0]] = 0
                    doNormalBB = True
                else:
                    doNormalBB = False
            
            if doNormalBB:
                ti = brentq(solveTi, -1/p[K[0]], 1, args=(Di, p, np.array(ai)), full_output=True)
                if ti[1].converged != True:
                    print('Oh no!')
                
                # Compute Ai
                for k in K:
                    Ai[k] = Aji(ai[k], p[k], ti[0])
            
            f = []
            for i in range(len(K)):
                f.append(p[i]*Ai[i])
            f = np.array(f)
            f = f*(f>0)
            
            if f.sum() > 0:
                LL += Di*np.log(f.sum()) - f.sum()
            elif Di > 0:
                LL+= -Di*1e7
            
            for k in K:
                if Ai[k] > 0:
                    LL += ai[k]*np.log(Ai[k]) - Ai[k]
                elif ai[k] > 0:
                    LL += -ai[k]*1e7
            
            if getA:
                AIJ.append(Ai)
    
    if getA:
        AIJ = np.array(AIJ)
        return -2*LL, AIJ
    
    return -2*LL

########################################################################
# Maximize likelihood

def getMaxLikelihood(hdata, hMX, startingPs,  plotFigure = False):
    nMCXtot = hMX[0].sum()
    nMCXe15 = hMX[1].sum()
    nMCXi15 = hMX[2].sum()
    nMCXe18 = hMX[3].sum()
    nMCXi18 = hMX[4].sum()

    # Set up Minuit
    def ll(nX, nE15, nI15, nE18, nI18):
        p0 = nX/nMCXtot
        p1 = nE15/nMCXe15
        p2 = nI15/nMCXi15
        p3 = nE18/nMCXe18
        p4 = nI18/nMCXi18
        return LogLikelihood(p0, p1, p2, p3, p4, hdata, hMX, False)

    logL = Minuit(ll, startingPs[0], startingPs[1], startingPs[2], startingPs[3], startingPs[4])
    #logL.tol = 1e-18
    logL.limits[0] = (0, None)
    logL.limits[1] = (0, None)
    logL.limits[2] = (0, None)
    logL.limits[3] = (0, None)
    logL.limits[4] = (0, None)

    startTime = time.time()

    # Solve
    logL.simplex()
    logL.strategy = 2
    logL.migrad()
    logL.hesse()

    values = logL.values

    # Print results
    print(values)
    print(logL.errors)
    print(logL.fval)
    print(logL.accurate)
    print('Elapsed time: ' + str(time.time() - startTime))

    if plotFigure:
        pvalues = values/np.array([nMCXtot, nMCXe15, nMCXi15, nMCXe18, nMCXi18])

        val, AIJ = LogLikelihood(pvalues[0], pvalues[1], pvalues[2], pvalues[3], pvalues[4], hdata, hMX, True)

        print(val)

        # Compute best fit histograms
        hBestFit = []
        for j in range(len(hMX)):
            hj = []
            for I in range(dthnBins):
                htemp = []
                for J in range(esumnBins):
                    htemp.append(AIJ[I*esumnBins + J][j]*pvalues[j])
                
                hj.append(np.array(htemp))
                
            hBestFit.append(np.array(hj))

        # Reshape
        hBestFit = np.array(hBestFit)

        # Plot marginal distributions
        fig = plt.figure(figsize=(42, 14), dpi=100)
        plt.subplot(131)
        for j in range(len(hMX)):
            if j == len(hMX) - 1:
                bottom = np.sum(hBestFit[:], axis=0)
                bottom = np.sum(bottom, axis=1)
                plt.plot(binsdatax[:-1], (bottom - np.sum(hdata, axis=1))/np.sqrt(np.sum(hdata, axis=1)), 'k--')

        plt.subplot(132)
        # make bar step
        plt.step(binsdatax[:-1], np.sum(hdata, axis=1), where='post', label='data', linewidth=2, color='k')
        # Stack MC fit
        bottom = np.sum(hBestFit[0:], axis=0)
        bottom = np.sum(bottom, axis=1)
        plt.bar(binsdatax[:-1], bottom, width=(binsdatax[1] - binsdatax[0]), alpha=0.5, label='MC BKG', align='edge')
        plt.bar(binsdatax[:-1], np.sum(hBestFit[0], axis=1), width=(binsdatax[1] - binsdatax[0]),bottom=bottom, alpha=0.5, label='MC X17', align='edge')

        plt.xlim(100, 180)
        plt.ylim(0, 3000*20/dthnBins)
        plt.legend()
        plt.xlabel('Relative angle [deg]')
        plt.grid()


        plt.subplot(133)
        # make bar step
        plt.step(binsdatay[:-1], np.sum(hdata, axis=0), where='post', label='data', linewidth=2, color='k')
        # Stack MC fit
        bottom = np.sum(hBestFit[0:], axis=0)
        bottom = np.sum(bottom, axis=0)
        plt.bar(binsdatay[:-1], bottom, width=(binsdatay[1] - binsdatay[0]), alpha=0.5, label='MC BKG', align='edge')
        plt.bar(binsdatay[:-1], np.sum(hBestFit[0], axis=0), width=(binsdatay[1] - binsdatay[0]),bottom=bottom, alpha=0.5, label='MC X17', align='edge')
        plt.xlabel('Energy sum [MeV]')
        plt.grid()

########################################################################
# Profile likelihood
def doProfileLL(startingPs, hdata, hMX, plotFigure = False):
    X = []
    Y = []
    nMCXtot = hMX[0].sum()
    nMCXe15 = hMX[1].sum()
    nMCXi15 = hMX[2].sum()
    nMCXe18 = hMX[3].sum()
    nMCXi18 = hMX[4].sum()
    def ll(nX, nE15, nI15, nE18, nI18):
        p0 = nX/nMCXtot
        p1 = nE15/nMCXe15
        p2 = nI15/nMCXi15
        p3 = nE18/nMCXe18
        p4 = nI18/nMCXi18
        return LogLikelihood(p0, p1, p2, p3, p4, hdata, hMX, False)
    
    for i in range(5):
        X.append(i*200)
        
        logL = Minuit(ll, X[-1], startingPs[1], startingPs[2], startingPs[3], startingPs[4])
        #logL.tol = 1e-18
        logL.limits[0] = (0, None)
        logL.limits[1] = (0, None)
        logL.limits[2] = (0, None)
        logL.limits[3] = (0, None)
        logL.limits[4] = (0, None)
        logL.fixed[0] = True

        startTime = time.time()

        # Solve
        logL.simplex()
        logL.strategy = 2
        logL.migrad()
        logL.hesse()
        
        Y.append(logL.fval)
    if plotFigure:
        plt.figure(figsize=(14, 14), dpi=100)
        plt.plot(X, Y, 'k--')
        plt.xlabel('X17 population')
        plt.ylabel(r'$-2\log{\mathcal{L}}$')
        plt.grid()
    return X, Y


########################################################################
# Significance
def computeSignificance(H0, H1, DOF):
    lratio = H0 - H1
    print('Likelihood ratio: ' + str(lratio))
    print('p-value: ' + str(1 - chi2.cdf(lratio, DOF)))
    print('Significance: ' + str(norm.ppf(1 - chi2.cdf(lratio, DOF))))
    return lratio, 1 - chi2.cdf(lratio, DOF), norm.ppf(1 - chi2.cdf(lratio, DOF))

########################################################################
# Main for testing
if __name__ == '__main__':
    # Get data and MC
    hMX, binsXMCx, binsXMCy = loadMC(MCFile)
    hdata, binsdatax, binsdatay = loadData(dataFile)
    startingPs = np.array([450, 37500, 27500, 135000, 50000])
    getMaxLikelihood(hdata, hMX, startingPs, plotFigure = True)
    doProfileLL(startingPs, hdata, hMX, plotFigure = True)