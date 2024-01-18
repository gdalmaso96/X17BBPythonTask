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
import SigLikX17, Ipc15LikX17, Ipc18LikX17, Epc15LikX17, Epc18LikX17
import numba as nb
from numba import jit


matplotlib.rcParams.update({'font.size': 35})
plt.rcParams['figure.constrained_layout.use'] = True

dataFile = 'X17MC2021.root'
#dataFile = 'X17MC2021_s0.root'
dataFile = 'data2023.root'
MCFile = 'X17reference.root'
MCFile = 'X17referenceRealistic.root'
MCFile = 'MC2023.root'
workDir = 'results/'

dthMin = 30
dthMax = 180
dthnBins = 15

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
def loadMC(fileName, workDir = '', ecodeCoding=0):
    with uproot.open(workDir + fileName + ':ntuple') as MC:
        esumMultiplier = 1
        try:
            x = MC.arrays(['imas', 'ecode', 'esum', 'dth'], library='np')
            if VariableSelection == 0:
                var = 'dth'
            elif VariableSelection == 1:
                var = 'imas'
        except:
            x = MC.arrays(['invm', 'ecode', 'esum', 'angle'], library='np')
            if VariableSelection == 0:
                var = 'angle'
            elif VariableSelection == 1:
                var = 'invm'
            esumMultiplier = 1e3
            
        if ecodeCoding == 1: # Hicham's coding
            vXMC    = x[var][x['ecode'] == 0]
            vI176MC = x[var][x['ecode'] == 1]
            vI179MC = x[var][x['ecode'] == 2]
            vI181MC = x[var][x['ecode'] == 3]
            vI146MC = x[var][x['ecode'] == 4]
            vI149MC = x[var][x['ecode'] == 5]
            vI151MC = x[var][x['ecode'] == 6]
            vE18MC  = x[var][x['ecode'] == 7]
            vE15MC  = x[var][x['ecode'] == 8]
            
            eXMC    = x['esum'][x['ecode'] == 0]*esumMultiplier
            eI176MC = x['esum'][x['ecode'] == 1]*esumMultiplier
            eI179MC = x['esum'][x['ecode'] == 2]*esumMultiplier
            eI181MC = x['esum'][x['ecode'] == 3]*esumMultiplier
            eI146MC = x['esum'][x['ecode'] == 4]*esumMultiplier
            eI149MC = x['esum'][x['ecode'] == 5]*esumMultiplier
            eI151MC = x['esum'][x['ecode'] == 6]*esumMultiplier
            eE18MC  = x['esum'][x['ecode'] == 7]*esumMultiplier
            eE15MC  = x['esum'][x['ecode'] == 8]*esumMultiplier
            
            # Create histograms
            hXMC, binsXMCx, binsXMCy     = np.histogram2d(vXMC, eXMC, bins=[dthnBins, esumnBins], range=[[dthMin, dthMax], [esumMin, esumMax]])
            hI176MC, binsI176MCx, binsI176MCy = np.histogram2d(vI176MC, eI176MC, bins=[dthnBins, esumnBins], range=[[dthMin, dthMax], [esumMin, esumMax]])
            hI179MC, binsI179MCx, binsI179MCy = np.histogram2d(vI179MC, eI179MC, bins=[dthnBins, esumnBins], range=[[dthMin, dthMax], [esumMin, esumMax]])
            hI181MC, binsI181MCx, binsI181MCy = np.histogram2d(vI181MC, eI181MC, bins=[dthnBins, esumnBins], range=[[dthMin, dthMax], [esumMin, esumMax]])
            hI146MC, binsI146MCx, binsI146MCy = np.histogram2d(vI146MC, eI146MC, bins=[dthnBins, esumnBins], range=[[dthMin, dthMax], [esumMin, esumMax]])
            hI149MC, binsI149MCx, binsI149MCy = np.histogram2d(vI149MC, eI149MC, bins=[dthnBins, esumnBins], range=[[dthMin, dthMax], [esumMin, esumMax]])
            hI151MC, binsI151MCx, binsI151MCy = np.histogram2d(vI151MC, eI151MC, bins=[dthnBins, esumnBins], range=[[dthMin, dthMax], [esumMin, esumMax]])
            hE18MC, binsE18MCx, binsE18MCy = np.histogram2d(vE18MC, eE18MC, bins=[dthnBins, esumnBins], range=[[dthMin, dthMax], [esumMin, esumMax]])
            hE15MC, binsE15MCx, binsE15MCy = np.histogram2d(vE15MC, eE15MC, bins=[dthnBins, esumnBins], range=[[dthMin, dthMax], [esumMin, esumMax]])
        
            hMX = np.array([hXMC, hI176MC, hI179MC, hI181MC, hI146MC, hI149MC, hI151MC, hE18MC, hE15MC])
        elif ecodeCoding == 0: # Fabrizio's coding
            vXMC   = x[var][x['ecode'] == 0]
            vE15MC = x[var][x['ecode'] == 1]
            vI15MC = x[var][x['ecode'] == 2]
            vE18MC = x[var][x['ecode'] == 3]
            vI18MC = x[var][x['ecode'] == 4]
            
            eXMC   = x['esum'][x['ecode'] == 0]*esumMultiplier
            eE15MC = x['esum'][x['ecode'] == 1]*esumMultiplier
            eI15MC = x['esum'][x['ecode'] == 2]*esumMultiplier
            eE18MC = x['esum'][x['ecode'] == 3]*esumMultiplier
            eI18MC = x['esum'][x['ecode'] == 4]*esumMultiplier
            
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
        try:
            x = data.arrays(['imas', 'esum', 'dth'], library='np')
            if VariableSelection == 0:
                var = 'dth'
            elif VariableSelection == 1:
                var = 'imas'
                
            vXdata   = x[var]
            eXdata   = x['esum']
        except:
            x = data.arrays(['invm', 'esum', 'angle'], library='np')
            if VariableSelection == 0:
                var = 'angle'
            elif VariableSelection == 1:
                var = 'invm'
                
            vXdata   = x[var]
            eXdata   = x['esum']*1e3
        
        # Create histograms
        hdata, binsdatax, binsdatay = np.histogram2d(vXdata, eXdata, bins=[dthnBins, esumnBins], range=[[dthMin, dthMax], [esumMin, esumMax]])
    
    return hdata, binsdatax, binsdatay

# Define Aji
# -- aji: MC ecode j, population of bin i
# -- pj: population j weight
# -- di: data population of bin i
# -- fi: MC prediction of population of bin i
@jit(nopython=True)
def Aji(aji, pj, ti):
    return aji/(1 + ti*pj)

# Define ti solver
# Ignore the null case for the moment
# -- di: data population of bin i
# -- p: array of pj
# -- ai: array of aji
@jit(nopython=True)
def solveTi(ti, di, p, ai):
    if ti == 1:
        return np.inf
    elif (ti == -1/p).any():
        return -np.inf
    A = di/(1 - ti)
    B = p*ai/(1 + ti*p)
    B = B.sum()
    return A - B


def LogLikelihood(p, hdata, hMX, getA=False, Kstart = 0):
    LL = 0
    AIJ = []
    TI = []
    p = np.array(p)
    K = np.argsort(p)[::-1]
    
    # Run through bins
    if VariableSelection == 0:
        nBins = dthnBins
    elif VariableSelection == 1:
        nBins = imasnBins
    
    for I in range(nBins):
        for J in range(esumnBins):
            ti = [0]
            Di = hdata[I][J]
            
            ai = []
            Ai = []
            for i in range(len(hMX)):
                ai.append(hMX[i][I][J])
                Ai.append(hMX[i][I][J])
            ai = np.array(ai)
            Ai = np.array(Ai)
            
            # Do null bins a la Beeston-Barlow
            if Kstart < len(K):
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
            
            if Kstart < len(K):
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
def getMaxLikelihood(hdata, hMX, binsdatax, binsdatay, startingPs,  plotFigure = False, doNullHyphotesis = False,  parametrizedX17 = False, DoMINOS = False, FixMass = False, fullParametrized = False, ecodeCoding = 0):
    massIndex = 5
    if ecodeCoding == 0:
        nMCXtot = hMX[0].sum()
        nMCXe15 = hMX[1].sum()
        nMCXi15 = hMX[2].sum()
        nMCXe18 = hMX[3].sum()
        nMCXi18 = hMX[4].sum()
    elif ecodeCoding == 1:
        nMCXtot = 1e5
        nMCXi176 = hMX[1].sum()
        nMCXi179 = hMX[2].sum()
        nMCXi181 = hMX[3].sum()
        nMCXi146 = hMX[4].sum()
        nMCXi149 = hMX[5].sum()
        nMCXi151 = hMX[6].sum()
        nMCXe18 = hMX[7].sum()
        nMCXe15 = hMX[8].sum()
        massIndex = 9
    
        print('nMCXtot: ' + str(nMCXtot))
        print('nMCXi176: ' + str(nMCXi176))
        print('nMCXi179: ' + str(nMCXi179))
        print('nMCXi181: ' + str(nMCXi181))
        print('nMCXi146: ' + str(nMCXi146))
        print('nMCXi149: ' + str(nMCXi149))
        print('nMCXi151: ' + str(nMCXi151))
        print('nMCXe18: ' + str(nMCXe18))
        print('nMCXe15: ' + str(nMCXe15))
    
    print('nData: ' + str(hdata.sum()))
    
    # Bin centers with mesh grid
    X, Y = np.meshgrid((binsdatax[:-1] + binsdatax[1:])/2, (binsdatay[:-1] + binsdatay[1:])/2)

    # Set up Minuit
    if fullParametrized and ecodeCoding == 0:
        fEpc15 = Epc15LikX17.AngleVSEnergySum(dthMin, dthMax, dthnBins, esumMin, esumMax, esumnBins)
        fIpc15 = Ipc15LikX17.AngleVSEnergySum(dthMin, dthMax, dthnBins, esumMin, esumMax, esumnBins)
        fEpc18 = Epc18LikX17.AngleVSEnergySum(dthMin, dthMax, dthnBins, esumMin, esumMax, esumnBins)
        fIpc18 = Ipc18LikX17.AngleVSEnergySum(dthMin, dthMax, dthnBins, esumMin, esumMax, esumnBins)
        hMX[1] = fEpc15.pdf(X, Y)*nMCXe15
        hMX[2] = fIpc15.pdf(X, Y)*nMCXi15
        hMX[3] = fEpc18.pdf(X, Y)*nMCXe18
        hMX[4] = fIpc18.pdf(X, Y)*nMCXi18
        def ll(nX, nE15, nI15, nE18, nI18, mX17):
            nMCXtot = nMCXtotParametrized
            hMX[0] = nMCXtot*SigLikX17.AngleVSEnergySum(X, Y, mX17, dthMin, dthMax, dthnBins, esumMin, esumMax, esumnBins, dthRes = 9.5, esumRes = 1.15)
            p0 = nX/nMCXtot
            p1 = nE15/nMCXe15
            p2 = nI15/nMCXi15
            p3 = nE18/nMCXe18
            p4 = nI18/nMCXi18
            return LogLikelihood([p0, p1, p2, p3, p4], hdata, hMX, False, Kstart = 10)
        
        logL = Minuit(ll, startingPs[0], startingPs[1], startingPs[2], startingPs[3], startingPs[4], startingPs[5])
        logL.limits[0] = (0, None)
        logL.limits[1] = (0, None)
        logL.limits[2] = (0, None)
        logL.limits[3] = (0, None)
        logL.limits[4] = (0, None)
        logL.limits[massIndex] = (15, 18.15)
        logL.fixed[0] = doNullHyphotesis
        logL.fixed[massIndex] = doNullHyphotesis + FixMass
    
    elif fullParametrized and ecodeCoding == 1:
        print('No PDFs are available for full parametrized X17 fit')
        return
        
    elif parametrizedX17:
        if ecodeCoding == 0:
            def ll(nX, nE15, nI15, nE18, nI18, mX17):
                nMCXtot = nMCXtotParametrized
                hMX[0] = nMCXtot*SigLikX17.AngleVSEnergySum(X, Y, mX17, dthMin, dthMax, dthnBins, esumMin, esumMax, esumnBins, dthRes = 9.5, esumRes = 1.15)
                p0 = nX/nMCXtot
                p1 = nE15/nMCXe15
                p2 = nI15/nMCXi15
                p3 = nE18/nMCXe18
                p4 = nI18/nMCXi18
                return LogLikelihood([p0, p1, p2, p3, p4], hdata, hMX, False, Kstart = 1)
            
            logL = Minuit(ll, startingPs[0], startingPs[1], startingPs[2], startingPs[3], startingPs[4], startingPs[5])
            logL.limits[0] = (0, None)
            logL.limits[1] = (0, None)
            logL.limits[2] = (0, None)
            logL.limits[3] = (0, None)
            logL.limits[4] = (0, None)
            logL.limits[massIndex] = (15, 18.15)
            logL.fixed[0] = doNullHyphotesis
            logL.fixed[massIndex] = doNullHyphotesis + FixMass
        elif ecodeCoding == 1:
            def ll(nX, nI176, nI179, nI181, nI146, nI149, nI151, nE18, nE15, mX17):
                nMCXtot = nMCXtotParametrized
                hMX[0] = nMCXtot*SigLikX17.AngleVSEnergySum(X, Y, mX17, dthMin, dthMax, dthnBins, esumMin, esumMax, esumnBins, dthRes = 9.5, esumRes = 1.15)
                p0 = nX/nMCXtot
                p1 = nI176/nMCXi176
                p2 = nI179/nMCXi179
                p3 = nI181/nMCXi181
                p4 = nI146/nMCXi146
                p5 = nI149/nMCXi149
                p6 = nI151/nMCXi151
                p7 = nE18/nMCXe18
                p8 = nE15/nMCXe15
                return LogLikelihood([p0, p1, p2, p3, p4, p5, p6, p7, p8], hdata, hMX, False, Kstart = 1)
            
            logL = Minuit(ll, startingPs[0], startingPs[1], startingPs[2], startingPs[3], startingPs[4], startingPs[5], startingPs[6], startingPs[7], startingPs[8], startingPs[9])
            logL.limits[0] = (0, None)
            logL.limits[1] = (0, None)
            logL.limits[2] = (0, None)
            logL.limits[3] = (0, None)
            logL.limits[4] = (0, None)
            logL.limits[5] = (0, None)
            logL.limits[6] = (0, None)
            logL.limits[7] = (0, None)
            logL.limits[8] = (0, None)
            logL.limits[massIndex] = (15, 18.15)
            logL.fixed[0] = doNullHyphotesis
            logL.fixed[massIndex] = doNullHyphotesis + FixMass
            
    else:
        if ecodeCoding == 0:
            def ll(nX, nE15, nI15, nE18, nI18):
                p0 = nX/nMCXtot
                p1 = nE15/nMCXe15
                p2 = nI15/nMCXi15
                p3 = nE18/nMCXe18
                p4 = nI18/nMCXi18
                return LogLikelihood([p0, p1, p2, p3, p4], hdata, hMX, False)
            
            startingPs = startingPs[:5]
            logL = Minuit(ll, startingPs[0], startingPs[1], startingPs[2], startingPs[3], startingPs[4])
            logL.limits[0] = (0, None)
            logL.limits[1] = (0, None)
            logL.limits[2] = (0, None)
            logL.limits[3] = (0, None)
            logL.limits[4] = (0, None)
            logL.fixed[0] = doNullHyphotesis
        elif ecodeCoding == 1:
            def ll(nX, nI176, nI179, nI181, nI146, nI149, nI151, nE18, nE15):
                p0 = nX/nMCXtot
                p1 = nI176/nMCXi176
                p2 = nI179/nMCXi179
                p3 = nI181/nMCXi181
                p4 = nI146/nMCXi146
                p5 = nI149/nMCXi149
                p6 = nI151/nMCXi151
                p7 = nE18/nMCXe18
                p8 = nE15/nMCXe15
                return LogLikelihood([p0, p1, p2, p3, p4, p5, p6, p7, p8], hdata, hMX, False)

            startingPs = startingPs[:9]
            logL = Minuit(ll, startingPs[0], startingPs[1], startingPs[2], startingPs[3], startingPs[4], startingPs[5], startingPs[6], startingPs[7], startingPs[8])
            logL.limits[0] = (0, None)
            logL.limits[1] = (0, None)
            logL.limits[2] = (0, None)
            logL.limits[3] = (0, None)
            logL.limits[4] = (0, None)
            logL.limits[5] = (0, None)
            logL.limits[6] = (0, None)
            logL.limits[7] = (0, None)
            logL.limits[8] = (0, None)
            logL.fixed[0] = doNullHyphotesis
    

    startTime = time.time()
    # Solve
    logL.hesse()
    
    #print(logL.values)
    #logL.print_level = 2
    initialFval = logL.fval
    logL.simplex(ncall=100000)
    logL.strategy = 1
    logL.migrad(ncall=100000, iterate=10)
    logL.hesse()
    print(initialFval)
    print(logL.fval)
    print(logL.valid)
    #print(logL.accurate)
    I = 0
    
    reFit = not logL.valid
    print('\n',reFit)
    print((reFit or logL.fval - initialFval > 1e-3) and I < 5)
    previousFval = logL.fval
    if (reFit or logL.fval - initialFval > 1e-3):
        if parametrizedX17 and not doNullHyphotesis:
            logL.fixed[0] = True
            logL.migrad(ncall=100000, iterate=10)
            logL.fixed[0] = False
            logL.fixed[massIndex] = True
            logL.migrad(ncall=100000, iterate=10)
            logL.fixed[0] = True
            logL.migrad(ncall=100000, iterate=10)
            logL.fixed[0] = False
            logL.fixed[massIndex] = True
            logL.migrad(ncall=100000, iterate=10)
            logL.fixed[massIndex] = False
        elif not doNullHyphotesis:
            logL.fixed[0] = True
            logL.migrad(ncall=100000, iterate=10)
            logL.fixed[0] = False
        logL.simplex(ncall=100000)
        logL.strategy = 1
        logL.migrad(ncall=100000, iterate=10)
        logL.hesse()
        print(initialFval)
        print(logL.fval)
        print(logL.valid)
        I = 0
        
        reFit = not logL.valid
        print('\n',reFit)
        print((reFit or logL.fval - initialFval > 1e-3) and I < 5)
        previousFval = logL.fval
    
    while ((reFit or logL.fval - initialFval > 1e-3) and I < 5):
        print('Elapsed time: ' + str(time.time() - startTime))
        print('Trying again')
        print('Start scan')
    
        for i in range(len(logL.values)):
            logL.limits[i] = (logL.values[i] - 5*np.sqrt(logL.values[i]), logL.values[i] + 5*np.sqrt(logL.values[i]))
            if logL.limits[i][0] < 0:
                logL.limits[i] = (0, logL.limits[i][1])
            if logL.limits[i][1] < 0:
                logL.limits[i] = (logL.limits[i][0], 1000)
        logL.scan(ncall = 10*(I + 2))
        logL.hesse()
        print(logL.fval)
        print(logL.valid)
        print(logL.values)
        for i in range(massIndex):
            logL.limits[i] = (0, None)
        logL.errors = np.array(logL.errors)*0.01
        logL.migrad(ncall = 1000000, iterate = 10)
        print(logL.fval)
        print(logL.valid)
        I += 1
        if I > 10:
            break
        if logL.fval < initialFval and logL.valid:
            break
        if logL.fval == previousFval:
            break
        previousFval = logL.fval
        reFit = not logL.valid
    logL.hesse()

    values = logL.values

    # Print results
    for i in range(len(values)):
        print(logL.parameters[i] + ': ' + str(values[i]) + ' +/- ' + str(logL.errors[i]))
    
    if ecodeCoding == 0:
        pvalues = values[:5]/np.array([nMCXtot, nMCXe15, nMCXi15, nMCXe18, nMCXi18])
    elif ecodeCoding == 1:
        pvalues = values[:9]/np.array([nMCXtot, nMCXi176, nMCXi179, nMCXi181, nMCXi146, nMCXi149, nMCXi151, nMCXe18, nMCXe15])
    print(pvalues.argsort()[::-1])
    print(logL.errors)
    print('Elapsed time: ' + str(time.time() - startTime))

    if plotFigure:
        if fullParametrized:
            mX17 = values[massIndex]
            nMCXtot = nMCXtotParametrized
            hMX[0] = nMCXtot*SigLikX17.AngleVSEnergySum(X, Y, mX17, dthMin, dthMax, dthnBins, esumMin, esumMax, esumnBins, dthRes = 9.5, esumRes = 1.15)
            hMX[1] = fEpc15.pdf(X, Y)*nMCXe15
            hMX[2] = fIpc15.pdf(X, Y)*nMCXi15
            hMX[3] = fEpc18.pdf(X, Y)*nMCXe18
            hMX[4] = fIpc18.pdf(X, Y)*nMCXi18
        elif parametrizedX17:
            mX17 = values[massIndex]
            nMCXtot = nMCXtotParametrized
            hMX[0] = nMCXtot*SigLikX17.AngleVSEnergySum(X, Y, mX17, dthMin, dthMax, dthnBins, esumMin, esumMax, esumnBins, dthRes = 9.5, esumRes = 1.15)
        
        if ecodeCoding == 0:
            pvalues = values[:5]/np.array([nMCXtot, nMCXe15, nMCXi15, nMCXe18, nMCXi18])
        elif ecodeCoding == 1:
            pvalues = values[:9]/np.array([nMCXtot, nMCXi176, nMCXi179, nMCXi181, nMCXi146, nMCXi149, nMCXi151, nMCXe18, nMCXe15])
        #pvalues = np.array([0, 21000, 1e3, 1.9e3, 1.9e3, 4.2e3, 2e3, 5.6e3, 1.2e3])/np.array([nMCXtot, nMCXi176, nMCXi179, nMCXi181, nMCXi146, nMCXi149, nMCXi151, nMCXe18, nMCXe15])

        if fullParametrized:
            val, AIJ, TI = LogLikelihood([pvalues[0], pvalues[1], pvalues[2], pvalues[3], pvalues[4]], hdata, hMX, True, Kstart=10)
        elif parametrizedX17:
            if ecodeCoding == 0:
                val, AIJ, TI = LogLikelihood([pvalues[0], pvalues[1], pvalues[2], pvalues[3], pvalues[4]], hdata, hMX, True, Kstart=1)
            elif ecodeCoding == 1:
                val, AIJ, TI = LogLikelihood([pvalues[0], pvalues[1], pvalues[2], pvalues[3], pvalues[4], pvalues[5], pvalues[6], pvalues[7], pvalues[8]], hdata, hMX, True, Kstart=1)
        else:
            if ecodeCoding == 0:
                val, AIJ, TI = LogLikelihood([pvalues[0], pvalues[1], pvalues[2], pvalues[3], pvalues[4]], hdata, hMX, True)
            elif ecodeCoding == 1:
                val, AIJ, TI = LogLikelihood([pvalues[0], pvalues[1], pvalues[2], pvalues[3], pvalues[4], pvalues[5], pvalues[6], pvalues[7], pvalues[8]], hdata, hMX, True)
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
                bottomMC = np.zeros((dthnBins, esumnBins))
                for k in range(1, massIndex):
                    bottomMC += hMX[k] * pvalues[k]
                
                plt.imshow((TI).transpose()[::-1], cmap=cm.coolwarm, extent=[binsdatax.min(), binsdatax.max(), binsdatay.min(), binsdatay.max()], aspect='auto')
                plt.grid()
                cbar = plt.colorbar(orientation='horizontal')
                cbar.set_label('ti')

                
                plt.xlabel('Relative angle, [deg]')
                plt.ylabel('Energy sum [MeV]')
                bottom = np.sum(bottom, axis=1)
                bottomMC = np.sum(bottomMC, axis=1)
        
        # Compute chi2 p-value
        # Compute lratio
        LRATIO = 0
        DOF = 0
        for I in range(dthnBins):
            for J in range(esumnBins):
                if np.sum(hBestFit[:, I, J]) > 0 and hdata[I][J] > 0:
                    mui = np.sum(hBestFit[:, I, J])
                    ni = hdata[I][J]
                    LRATIO += ni*np.log(ni/mui) - ni + mui
        
        CHI2 = LRATIO
        print('\n--- Goodness of fit ---')
        print('Chi2: ' + str(CHI2))
        DOF = len(np.array(logL.fixed)[np.array(logL.fixed) == False])
        print('DOF: ' + str(DOF))
        print('p-value: ' + str(chi2.sf(CHI2, DOF)))
        print('sigma: ' + str(norm.isf(chi2.sf(CHI2, DOF)*0.5)))

        ax = plt.subplot(132)
        plt.stairs( np.sum(hdata, axis=1), binsdatax,label='data', linewidth=8, color='k')
        
        # Labels for background contributions
        BKGlabels = []
        if ecodeCoding == 0:
            BKGlabels = ['MC e15', 'MC i15', 'MC e18', 'MC i18']
        elif ecodeCoding == 1:
            BKGlabels = ['MC i17.6', 'MC i17.9', 'MC i18.1', 'MC i14.6', 'MC i14.9', 'MC i15.1', 'MC e18', 'MC e15']
        
        # Stack MC fit
        bottom = np.sum(hBestFit[1:], axis=0)
        bottom = np.sum(bottom, axis=1)
        plt.bar(binsdatax[:-1], bottom, width=(binsdatax[1] - binsdatax[0]), label='MC BKG', align='edge', color=cm.coolwarm(0), alpha=0.5)
        plt.stairs(bottom, binsdatax, linewidth=8, color=cm.coolwarm(0))
        plt.bar(binsdatax[:-1], np.sum(hBestFit[0], axis=1), width=(binsdatax[1] - binsdatax[0]),bottom=bottom, label='MC X17', align='edge', color=cm.coolwarm(0.99), alpha=0.5)
        plt.stairs(np.sum(hBestFit[0], axis=1), binsdatax, linewidth=8, color=cm.coolwarm(0.99))
        
        # Plot single contributions
        for k in range(1, massIndex):
            htemp = np.sum(hMX[k]*pvalues[k], axis=1)
            #plt.bar(binsdatax[:-1], htemp, width=(binsdatax[1] - binsdatax[0]), label=BKGlabels[k-1], align='edge', alpha=0.5, color='C%d'%k)
            plt.stairs(htemp, binsdatax, linewidth=8, color='C%d'%k, label=BKGlabels[k-1])
        
        # Add points end errorbars for data
        plt.errorbar((binsdatax[:-1] + binsdatax[1:])/2, np.sum(hdata, axis=1), yerr=np.sqrt(np.sum(hdata, axis=1)), fmt='o', color='k', label='data', markersize=10, linewidth=8)
        
        #plt.xlim(60, 180)
        #plt.ylim(0, 3000*20/dthnBins)
        plt.yscale('log')
        plt.legend()
        plt.xlabel('Relative angle [deg]')
        ax.set_xticklabels([])
        plt.grid()
        
        bounds = [0.3775, 0.05, 0.2925, 0.125]
        bottom = np.sum(hBestFit[:], axis=0)
        bottom = np.sum(bottom, axis=1)
        pull = (np.sum(hdata, axis=1) - bottom)/np.sqrt(bottom)
        bottom = np.sqrt(bottom)
        plt.gcf().add_axes(bounds)
        plt.bar(binsdatax[:-1], pull, width=(binsdatax[1] - binsdatax[0]), alpha=0.5, label='MC X17', align='edge', color=cm.coolwarm(0.99))
        #plt.xlim(60, 180)
        plt.grid()
        plt.xlabel('Relative angle [deg]')
        plt.ylabel('Pull')
        plt.ylim(-5.5, 5.5)


        ax = plt.subplot(133)
        plt.stairs(np.sum(hdata, axis=0), binsdatay, label='data', linewidth=8, color='k')
        
        # Stack MC fit
        bottom = np.sum(hBestFit[1:], axis=0)
        bottom = np.sum(bottom, axis=0)
        plt.bar(binsdatay[:-1], bottom, width=(binsdatay[1] - binsdatay[0]), alpha=0.5, label='MC BKG', align='edge', color=cm.coolwarm(0))
        plt.stairs(bottom, binsdatay, linewidth=8, color=cm.coolwarm(0))
        plt.bar(binsdatay[:-1], np.sum(hBestFit[0], axis=0), width=(binsdatay[1] - binsdatay[0]),bottom=bottom, alpha=0.5, label='MC X17', align='edge', color=cm.coolwarm(0.99))
        plt.stairs(np.sum(hBestFit[0], axis=0), binsdatay, linewidth=8, color=cm.coolwarm(0.99))
        plt.yscale('log')
        ax.set_xticklabels([])
        plt.grid()
        
        # Plot single contributions
        for k in range(1, massIndex):
            htemp = np.sum(hMX[k]*pvalues[k], axis=0)
            #plt.bar(binsdatay[:-1], htemp, width=(binsdatay[1] - binsdatay[0]), label=BKGlabels[k-1], align='edge', alpha=0.5, color='C%d'%k)
            plt.stairs(htemp, binsdatay, linewidth=8, color='C%d'%k, label=BKGlabels[k-1])
            
        # Add points end errorbars for data
        plt.errorbar((binsdatay[:-1] + binsdatay[1:])/2, np.sum(hdata, axis=0), yerr=np.sqrt(np.sum(hdata, axis=0)), fmt='o', color='k', label='data', markersize=10, linewidth=8)
        
        
        bounds = [0.70625, 0.05, 0.2925, 0.125]
        bottom = np.sum(hBestFit[:], axis=0)
        bottom = np.sum(bottom, axis=0)
        pull = (np.sum(hdata, axis=0) - bottom)/np.sqrt(bottom)
        bottom = np.sqrt(bottom)
        ax = plt.gcf().add_axes(bounds)
        plt.bar(binsdatay[:-1], pull, width=(binsdatay[1] - binsdatay[0]), alpha=0.5, label='MC X17', align='edge', color=cm.coolwarm(0.99))
        plt.grid()
        plt.xlabel('Energy sum [MeV]')
        plt.ylim(-5.5, 5.5)
        
        
        if not doNullHyphotesis:
            plt.savefig('X17Fit.png', bbox_inches='tight')
        else:
            plt.savefig('X17FitNull.png', bbox_inches='tight')
        
        # Plot minos matrix
        if DoMINOS:
            fig.clf()
            fix, ax = logL.draw_mnmatrix(cl=[0.68, 0.9], figsize=(42, 42))
            
    
    
    return values, logL.errors, logL.fval, logL.valid

########################################################################
# Profile likelihood
def doProfileLL(startingPs, hdata, hMX, plotFigure = False, ecodeCoding = 0):
    X = []
    Y = []
    massIndex = 5
    if ecodeCoding == 0:
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
            return LogLikelihood([p0, p1, p2, p3, p4], hdata, hMX, False)
    elif ecodeCoding == 1:
        nMCXtot = 1e5
        nMCXi176 = hMX[1].sum()
        nMCXi179 = hMX[2].sum()
        nMCXi181 = hMX[3].sum()
        nMCXi146 = hMX[4].sum()
        nMCXi149 = hMX[5].sum()
        nMCXi151 = hMX[6].sum()
        nMCXe18 = hMX[7].sum()
        nMCXe15 = hMX[8].sum()
        def ll(nX, nI176, nI179, nI181, nI146, nI149, nI151, nE18, nE15):
            p0 = nX/nMCXtot
            p1 = nI176/nMCXi176
            p2 = nI179/nMCXi179
            p3 = nI181/nMCXi181
            p4 = nI146/nMCXi146
            p5 = nI149/nMCXi149
            p6 = nI151/nMCXi151
            p7 = nE18/nMCXe18
            p8 = nE15/nMCXe15
            return LogLikelihood([p0, p1, p2, p3, p4, p5, p6, p7, p8], hdata, hMX, False)
        massIndex = 9
    for i in range(5):
        X.append(i*200)
        
        logL = Minuit(ll, X[-1], startingPs[1], startingPs[2], startingPs[3], startingPs[4])
        #logL.tol = 1e-18
        for k in range(massIndex):
            logL.limits[k] = (0, None)
        logL.fixed[0] = True

        startTime = time.time()

        # Solve
        logL.simplex()
        #logL.strategy = 2
        logL.migrad(ncall=100000)
        logL.hesse()
        
        Y.append(logL.fval)
    if plotFigure:
        plt.figure(figsize=(14, 14), dpi=100)
        plt.plot(X, Y, 'k--')
        plt.xlabel('X17 population')
        plt.ylabel(r'$-2\log{\mathcal{L}}$')
        plt.grid()
        plt.show()
    return X, Y


########################################################################
# Significance
def computeSignificance(H0, H1, DOF, Ncrossing = 0.44, c0 = 1, parametrizedX17 = False):
    lratio = H0 - H1
    lratio = lratio*(lratio > 0) + lratio*(lratio < -1e-3)
    pvalue = chi2.sf(lratio, DOF)
    if parametrizedX17:
        pvalue += Ncrossing*np.exp(-0.5*(lratio - c0))*(lratio/c0)**((DOF - 1)*0.5)
        #pvalue += Ncrossing*chi2.sf(lratio, DOF + 1) 
    pvalue = pvalue*(pvalue < 1) + 1*(pvalue >= 1)
    sigma = norm.isf(pvalue*0.5)
    #print('Likelihood ratio: ' + str(lratio))
    #print('p-value: ' + str(pvalue))
    #print('Significance: ' + str(sigma))
    return lratio, pvalue, sigma

########################################################################
# Main for testing
if __name__ == '__main__':
    ecodeCoding = 1
    # Get data and MC
    hMX, binsXMCx, binsXMCy = loadMC(MCFile, workDir, ecodeCoding=ecodeCoding)
    hdata, binsdatax, binsdatay = loadData(dataFile, workDir)
    startingPs = np.array([450, 37500, 27500, 135000, 50000, 17])
    #H1 = getMaxLikelihood(hdata, hMX, binsdatax, binsdatay, startingPs,  plotFigure = True, doNullHyphotesis = False, parametrizedX17 = True, DoMINOS = False, fullParametrized = False, FixMass = False)
    startingPs = np.array([0, 37500, 27500, 135000, 50000, 17])
    startingPs = np.array([0, 100000, 100000, 100000, 50000, 50000, 50000, 50000, 50000, 17])
    startingPs = np.array([0, 21000, 1e3, 1.9e3, 1.9e3, 4.2e3, 2e3, 5.6e3, 1.2e3, 17])
    H0 = getMaxLikelihood(hdata, hMX, binsdatax, binsdatay, startingPs,  plotFigure = True, doNullHyphotesis = True,  parametrizedX17 = False, DoMINOS = False, fullParametrized = False, ecodeCoding=1)
    
    print('\n--- Hypothesis test ---')
    print(computeSignificance(H0[2], H1[2], 2, parametrizedX17=False))
    #doProfileLL(startingPs, hdata, hMX, plotFigure = True)
