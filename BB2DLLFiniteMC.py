# This routine fits a 2D histogram with the Beeston-Barlow Likelihood
# The data and MC are a complete X17 production

import uproot
import numpy as np
from scipy.integrate import quad
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.optimize import brentq, differential_evolution
from iminuit import Minuit
import matplotlib
from scipy.stats import chi2, norm
import time
import SigLikX17

matplotlib.rcParams.update({'font.size': 35})
plt.rcParams['figure.constrained_layout.use'] = True

dataFile = 'X17MC2021.root'
#dataFile = 'X17MC2021_s1.root'
MCFile = 'X17reference.root'
#MCFile = 'X17referenceRealistic.root'
workDir = 'results/'

dthMin = 20
dthMax = 180
dthnBins = 20

esumMin = 10
esumMax = 24
esumnBins = 14

imasMin = 12
imasMax = 20
imasnBins = 40

nMCXtotParametrized = 1e9

VariableSelection = 0 # 0: dth, 1: imas

startingPs = np.array([450, 37500, 27500, 135000, 50000])

# Import MC histogram
def loadMC(fileName, workDir = ''):
    with uproot.open(workDir + fileName + ':ntuple') as MC:
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
def loadData(fileName, workDir = ''):
    with uproot.open(workDir + fileName + ':ntuple') as data:
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

def LogLikelihood(p0, p1, p2, p3, p4, hdata, hMX, getA=False, Kstart = 0):
    LL = 0
    AIJ = []
    TI = []
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
                    ti = [-1/p[K[0]]]
            
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
            
            for k in range(Kstart, len(K)):
                if Ai[k] > 0:
                    LL += ai[k]*np.log(Ai[k]) - Ai[k]
                elif ai[k] > 0:
                    LL += -ai[k]*1e7
            
            if getA:
                AIJ.append(Ai)
                TI.append(ti[0])
    
    if getA:
        AIJ = np.array(AIJ)
        TI = np.array(TI)
        return -2*LL, AIJ, TI
    
    return -2*LL

########################################################################
# Toy MC
def sampleToyMC(hMXtemp, SEED = 0):
    np.random.seed(SEED)
    hToyMC = []
    for i in range(len(hMXtemp)):
        hTemp = []
        for j in range(len(hMXtemp[i])):
            if hMXtemp[i][j] > 0:
                hTemp.append(np.random.poisson(hMXtemp[i][j]))
            else:
                hTemp.append(0)
        hToyMC.append(np.array(hTemp))
    return np.array(hToyMC)

# Do FC point

########################################################################
# Maximize likelihood

def getMaxLikelihood(hdata, hMX, binsdatax, binsdatay, startingPs,  plotFigure = False, doNullHyphotesis = False,  parametrizedX17 = False, doDEConvergenceOnly = False):
    nMCXtot = hMX[0].sum()
    nMCXe15 = hMX[1].sum()
    nMCXi15 = hMX[2].sum()
    nMCXe18 = hMX[3].sum()
    nMCXi18 = hMX[4].sum()
    
    # Bin centers with mesh grid
    X, Y = np.meshgrid((binsdatax[:-1] + binsdatax[1:])/2, (binsdatay[:-1] + binsdatay[1:])/2)

    # Set up Minuit
    if parametrizedX17:
        def ll(nX, nE15, nI15, nE18, nI18, mX17):
            nMCXtot = nMCXtotParametrized
            hMX[0] = nMCXtot*SigLikX17.AngleVSEnergySum(X, Y, mX17, dthMin, dthMax, dthnBins, esumMin, esumMax, esumnBins, dthRes = 9.5, esumRes = 1.15)
            p0 = nX/nMCXtot
            p1 = nE15/nMCXe15
            p2 = nI15/nMCXi15
            p3 = nE18/nMCXe18
            p4 = nI18/nMCXi18
            return LogLikelihood(p0, p1, p2, p3, p4, hdata, hMX, False, Kstart = 1)
        
        logL = Minuit(ll, startingPs[0], startingPs[1], startingPs[2], startingPs[3], startingPs[4], startingPs[5])
        logL.limits[0] = (0, None)
        logL.limits[1] = (0, None)
        logL.limits[2] = (0, None)
        logL.limits[3] = (0, None)
        logL.limits[4] = (0, None)
        logL.limits[5] = (15, 18.15)
        logL.fixed[0] = doNullHyphotesis
        logL.fixed[5] = doNullHyphotesis
    else:
        def ll(nX, nE15, nI15, nE18, nI18):
            p0 = nX/nMCXtot
            p1 = nE15/nMCXe15
            p2 = nI15/nMCXi15
            p3 = nE18/nMCXe18
            p4 = nI18/nMCXi18
            return LogLikelihood(p0, p1, p2, p3, p4, hdata, hMX, False)
        
        startingPs = startingPs[:5]
        logL = Minuit(ll, startingPs[0], startingPs[1], startingPs[2], startingPs[3], startingPs[4])
        #logL.tol = 1e-18
        logL.limits[0] = (0, None)
        logL.limits[1] = (0, None)
        logL.limits[2] = (0, None)
        logL.limits[3] = (0, None)
        logL.limits[4] = (0, None)
        logL.fixed[0] = doNullHyphotesis
    

    startTime = time.time()
    DEtolerance = 1e-6
    DEpopsize = 20
    if doDEConvergenceOnly:
        DEtolerance = 1e-7
        DEpopsize = 20
    # Find starting point with differential evolution
    bounds = []
    for sp,fix in zip(startingPs, logL.fixed):
        if fix:
            bounds.append((sp, sp))
        elif sp == 0:
            bounds.append((0, 100))
        else:
            bounds.append((sp*0.9, sp*1.1))
    if parametrizedX17:
        res = differential_evolution(lambda x: ll(x[0], x[1], x[2], x[3], x[4], x[5]), bounds, seed = int(time.time()), x0=startingPs, popsize=DEpopsize, updating='immediate', tol=DEtolerance, disp=True)
    else:
        res = differential_evolution(lambda x: ll(x[0], x[1], x[2], x[3], x[4]), bounds, seed = int(time.time()), x0=startingPs, popsize=DEpopsize, updating='immediate', tol=DEtolerance, disp=True)
    logL.values = res.x
    #if parametrizedX17 and not doNullHyphotesis:
    #    logL.values[0] = startingPs[0]
    #    logL.values[5] = startingPs[5]
    print('Found starting point:', res.x, res.fun)
    # Solve
    #logL.simplex()
    #logL.strategy = 2
    #logL.tol = 1e-18
    #logL.errors = res.x*0.1
    logL.migrad(ncall=100000)
    logL.hesse()
    print(logL.fval)
    print(logL.accurate)
    initialFval = res.fun
    I = 0
    if doDEConvergenceOnly and logL.fval - initialFval > 1e-3:
        logL.values = res.x
        logL.hesse()
    
    reFit = not logL.accurate
    #if logL.accurate == False:
    #    reFit = True
    #    for val, limits in zip(logL.values, logL.limits):
    #        if val == limits[0] or val == limits[1]:
    #            reFit = False
    #            break
    
    while ((reFit or logL.fval - initialFval > 1e-3) and I < 5 and not doDEConvergenceOnly):
        print('Elapsed time: ' + str(time.time() - startTime))
        print('Trying again')
        # Find starting point with differential evolution
        bounds = []
        for sp, fix in zip(startingPs, logL.fixed):
            if fix:
                bounds.append((sp, sp))
            elif sp == 0:
                bounds.append((0, 100))
            else:
                bounds.append((sp*0.5, sp*1.5))
        if parametrizedX17:
            res = differential_evolution(lambda x: ll(x[0], x[1], x[2], x[3], x[4], x[5]), bounds, seed = int(time.time()), maxiter = 1000, tol=DEtolerance, popsize=DEpopsize*(I+1), disp=True)
        else:
            bounds = bounds[:5]
            res = differential_evolution(lambda x: ll(x[0], x[1], x[2], x[3], x[4]), bounds, seed = int(time.time()), maxiter = 1000, tol=DEtolerance, popsize=DEpopsize*(I+1), disp=True)
        print('Elapsed time: ' + str(time.time() - startTime))
        print('Found starting point:', res.x, res.fun)
        logL.values = res.x
        #logL.simplex()
        #logL.strategy = 2
        #logL.tol = 1e-18
        logL.migrad(ncall = 1000000)
        print(logL.fval)
        print(logL.accurate)
        I += 1
        if I > 10:
            break
        if logL.fval < initialFval and logL.accurate:
            break
    logL.hesse()

    values = logL.values

    # Print results
    print(values)
    pvalues = values[:5]/np.array([nMCXtot, nMCXe15, nMCXi15, nMCXe18, nMCXi18])
    print(pvalues.argsort()[::-1])
    print(logL.errors)
    print('Elapsed time: ' + str(time.time() - startTime))

    if plotFigure:
        if parametrizedX17:
            mX17 = values[5]
            nMCXtot = nMCXtotParametrized
            hMX[0] = nMCXtot*SigLikX17.AngleVSEnergySum(X, Y, mX17, dthMin, dthMax, dthnBins, esumMin, esumMax, esumnBins, dthRes = 9.5, esumRes = 1.15)
        pvalues = values[:5]/np.array([nMCXtot, nMCXe15, nMCXi15, nMCXe18, nMCXi18])

        if parametrizedX17:
            val, AIJ, TI = LogLikelihood(pvalues[0], pvalues[1], pvalues[2], pvalues[3], pvalues[4], hdata, hMX, True, Kstart=1)
        else:
            val, AIJ, TI = LogLikelihood(pvalues[0], pvalues[1], pvalues[2], pvalues[3], pvalues[4], hdata, hMX, True)
        TI = TI.reshape((dthnBins, esumnBins))

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
        fig = plt.figure(figsize=(42, 16), dpi=100)
        ax = fig.add_subplot(131)
        for j in range(len(hMX)):
            if j == len(hMX) - 1:
                bottom = np.sum(hBestFit[1:], axis=0)
                bottomMC = hMX[1] * pvalues[1] + hMX[2] * pvalues[2] + hMX[3] * pvalues[3] + hMX[4] * pvalues[4] 
                
                #plt.imshow(((bottom - bottomMC)/bottom).transpose()[::-1], cmap=cm.coolwarm, extent=[binsdatax.min(), binsdatax.max(), binsdatay.min(), binsdatay.max()], aspect='auto')
                plt.imshow((TI).transpose()[::-1], cmap=cm.coolwarm, extent=[binsdatax.min(), binsdatax.max(), binsdatay.min(), binsdatay.max()], aspect='auto')
                plt.grid()
                cbar = plt.colorbar(orientation='horizontal')
                cbar.set_label('ti')
                #cbar.set_ticks(np.linspace(0, 2, num=10))

                
                plt.xlabel('Relative angle, [deg]')
                plt.ylabel('Energy sum [MeV]')
                bottom = np.sum(bottom, axis=1)
                bottomMC = np.sum(bottomMC, axis=1)
                #plt.plot(binsdatax[:-1], (bottom - np.sum(hdata, axis=1))/np.sqrt(np.sum(hdata, axis=1)), 'k--')

        plt.subplot(132)
        # make bar step
        plt.stairs( np.sum(hdata, axis=1), binsdatax,label='data', linewidth=8, color='k')
        # Stack MC fit
        bottom = np.sum(hBestFit[1:], axis=0)
        bottom = np.sum(bottom, axis=1)
        plt.bar(binsdatax[:-1], bottom, width=(binsdatax[1] - binsdatax[0]), label='MC BKG', align='edge', color=cm.coolwarm(0), alpha=0.5)
        plt.stairs(bottom, binsdatax, linewidth=8, color=cm.coolwarm(0))
        plt.bar(binsdatax[:-1], np.sum(hBestFit[0], axis=1), width=(binsdatax[1] - binsdatax[0]),bottom=bottom, label='MC X17', align='edge', color=cm.coolwarm(0.99), alpha=0.5)
        plt.stairs(np.sum(hBestFit[0], axis=1), binsdatax, linewidth=8, color=cm.coolwarm(0.99))

        plt.xlim(100, 180)
        plt.ylim(0, 3000*20/dthnBins)
        plt.legend()
        plt.xlabel('Relative angle [deg]')
        plt.grid()


        plt.subplot(133)
        # make bar step
        plt.stairs(np.sum(hdata, axis=0), binsdatay, label='data', linewidth=8, color='k')
        # Stack MC fit
        bottom = np.sum(hBestFit[1:], axis=0)
        bottom = np.sum(bottom, axis=0)
        plt.bar(binsdatay[:-1], bottom, width=(binsdatay[1] - binsdatay[0]), alpha=0.5, label='MC BKG', align='edge', color=cm.coolwarm(0))
        plt.stairs(bottom, binsdatay, linewidth=8, color=cm.coolwarm(0))
        plt.bar(binsdatay[:-1], np.sum(hBestFit[0], axis=0), width=(binsdatay[1] - binsdatay[0]),bottom=bottom, alpha=0.5, label='MC X17', align='edge', color=cm.coolwarm(0.99))
        plt.stairs(np.sum(hBestFit[0], axis=0), binsdatay, linewidth=8, color=cm.coolwarm(0.99))
        plt.yscale('log')
        plt.xlabel('Energy sum [MeV]')
        plt.grid()
        if not doNullHyphotesis:
            plt.savefig('X17Fit.png', bbox_inches='tight')
        else:
            plt.savefig('X17FitNull.png', bbox_inches='tight')
    
    
    return values, logL.errors, logL.fval, logL.accurate

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
        #logL.strategy = 2
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
    pvalue = chi2.sf(lratio, DOF)
    sigma = norm.isf(pvalue*0.5)
    print('Likelihood ratio: ' + str(lratio))
    print('p-value: ' + str(pvalue))
    print('Significance: ' + str(sigma))
    return lratio, pvalue, sigma

########################################################################
# Main for testing
if __name__ == '__main__':
    # Get data and MC
    hMX, binsXMCx, binsXMCy = loadMC(MCFile, workDir)
    hdata, binsdatax, binsdatay = loadData(dataFile, workDir)
    startingPs = np.array([450, 37500, 27500, 135000, 50000, 17])
    H1 = getMaxLikelihood(hdata, hMX, binsdatax, binsdatay, startingPs,  plotFigure = True, doNullHyphotesis = False, parametrizedX17 = True)
    startingPs = np.array([0, 37500, 27500, 135000, 50000, 17])
    H0 = getMaxLikelihood(hdata, hMX, binsdatax, binsdatay, startingPs,  plotFigure = True, doNullHyphotesis = True,  parametrizedX17 = True)
    computeSignificance(H0[2], H1[2], 2)
    #doProfileLL(startingPs, hdata, hMX, plotFigure = True)
