# Test estimators from BB2DLLFiniteMC.py
from sample2D import sampleMass
import numpy as np
import BB2DLLFiniteMC
# add argument parsin
import argparse
from time import time
from pathlib import Path
import SigLikX17
from scipy.interpolate import interp1d
import matplotlib 
from matplotlib import pyplot as plt
from matplotlib import cm
matplotlib.rcParams.update({'font.size': 35})

# Null signal profile python3 testEstimators.py --seed 0 --referenceFile X17referenceRealistic.root --profileLikelihood True --profileLikelihood2D True --prefix bins20x14CurrentStatisticsParametrizedNullSig --parametrizeX17 True --dataFile X17MC2021_s0 --mX17plMin 15 --numberPL 15

# Background test: python3 testEstimators.py --sampleHistograms True --posterioriFC True --ecodeCoding 1 --dataFile data2023 --referenceFile MC2023.root --parametrizeX17 True --BKGonly True --nSamples 100 --prefix BKGonlyFractionside

def argparser():
    parser = argparse.ArgumentParser(description='Test estimators from 2DLLFiniteMC.py')
    parser.add_argument('-n', '--nSamples', type=int, default=1000, help='Number of samples to draw')
    parser.add_argument('-s', '--seed', type=int, default=0, help='Random seed')
    parser.add_argument('-w', '--workDir', type=str, default='/Users/giovanni/PhD/Analysis/X17BBPythonTask/results/', help='Working directory')
    parser.add_argument('-r', '--reset', type=bool, help='Replace results file', default=False)
    parser.add_argument('-rf', '--referenceFile', type=str, default='X17reference.root', help='Reference MC file')
    parser.add_argument('-df', '--dataFile', type=str, default='X17MC2021', help='Data file')
    parser.add_argument('-p', '--prefix', type=str, default='BBstandard', help='Prefix for analysis')
    parser.add_argument('-rMC', '--reuseMC', type=bool, default=False, help='Reuse ToyMC produced previously')
    parser.add_argument('-pX17', '--parametrizeX17', type=bool, default=False, help='Use parametrized X17 template')
    parser.add_argument('-v', '--varyReference', type=bool, default=False, help='Sample reference MC at each iteration')
    parser.add_argument('-ts', '--ToySample', type=bool, default=False, help='Produce ToyMC sample')
    parser.add_argument('-nt', '--numberToys', type=int, default=1000, help='Number of toys to produce')
    parser.add_argument('-nX17', '--numberX17', type=float, default=450, help='Number of X17 events to produce in ToyMC')
    parser.add_argument('-mX17', '--massX17', type=float, default=17, help='Mass of X17 events to produce in ToyMC')
    parser.add_argument('-pFC', '--posterioriFC', type=bool, default=False, help='Use a posteriori FC')
    parser.add_argument('-rFC', '--resetFC', type=bool, default=False, help='Replace FC file')
    parser.add_argument('-st', '--saveToy', type=bool, default=False, help='Save ToyMC')
    parser.add_argument('-pT', '--plotToy', type=bool, default=False, help='Plot ToyMC')
    parser.add_argument('-pLL', '--profileLikelihood', type=bool, default=False, help='Use profile likelihood')
    parser.add_argument('-pLL2D', '--profileLikelihood2D', type=bool, default=False, help='Use profile likelihood')
    parser.add_argument('--nX17plMin', type=float, default=0, help='Minimum number of X17 events to scan')
    parser.add_argument('--nX17plMax', type=float, default=900, help='Maximum number of X17 events to scan')
    parser.add_argument('--mX17plMin', type=float, default=16, help='Minimum mass of X17 events to scan')
    parser.add_argument('--mX17plMax', type=float, default=18, help='Maximum mass of X17 events to scan')
    parser.add_argument('-nPL', '--numberPL', type=int, default=11, help='Number of points to scan profile likelihood')
    parser.add_argument('-pG', '--globalPValue', type=bool, default=False, help='Compute global p-value')
    parser.add_argument('-pGmin', '--pvalueMin', type=float, default=15, help='Minimum mass to scan')
    parser.add_argument('-pGmax', '--pvalueMax', type=float, default=18.15, help='Maximum mass to scan')
    parser.add_argument('-pGc0', '--pvalueC0', type=float, default=0.05, help='Threshold for global p-value')
    parser.add_argument('-ec', '--ecodeCoding', type=int, default=0, help='Use ecode coding: 0 from Fabrizio, 1 from Hicham')
    parser.add_argument('-sp', '--sampleHistograms', type=bool, default=False, help='Sample histograms to test estimators')
    parser.add_argument('-bO', '--BKGonly', type=bool, default=False, help='Test BKG only')
    #return parser.parse_known_args()
    return parser.parse_args()

def globalPvalue(i, workDir, referenceFile, nX17Toy, parametrizeX17, pvalueC0, MIN, MAX, prefix, startTime, numberPL):
    # Get template
    MCFile = f'{referenceFile}'
    hMX, binsdatax, binsdatay = BB2DLLFiniteMC.loadMC(MCFile, workDir = workDir)
    
    nMCXtot = hMX[0].sum()
    nMCXe15 = hMX[1].sum()
    nMCXi15 = hMX[2].sum()
    nMCXe18 = hMX[3].sum()
    nMCXi18 = hMX[4].sum()
    startingPs = np.array([0, 37500, 27500, 135000, 50000, 17], dtype=float)
    p = np.array([nX17Toy/nMCXtot, startingPs[1]/nMCXe15, startingPs[2]/nMCXi15, startingPs[3]/nMCXe18, startingPs[4]/nMCXi18])
    htemp = hMX[1]*p[1] + hMX[2]*p[2] + hMX[3]*p[3] + hMX[4]*p[4]
    print('Toy number: ', i)
    # Sample data ToyMC
    hdata = BB2DLLFiniteMC.sampleToyMC(htemp, SEED + i)

    # Sample reference ToyMCs
    hToyMC = []
    for j in range(len(hMX)):
        # Do not sample X17 if parametrized
        if j == 0 and parametrizeX17:
            hToyMC.append(0*hMX[j])
        else:
            hToyMC.append(BB2DLLFiniteMC.sampleToyMC(hMX[j], SEED + i))
    
    # Get null hypothesis
    startingPs = np.array([0, 37500, 27500, 135000, 50000, 17], dtype=float)
    valuesH0, errorsH0, fvalH0, validH0 = BB2DLLFiniteMC.getMaxLikelihood(hdata, hToyMC, binsdatax, binsdatay, startingPs, plotFigure = False, doNullHyphotesis=True, parametrizedX17 = parametrizeX17)
    
    # Create grid to scan profile likelihood
    massX17Scan = np.linspace(MIN, MAX, numberPL)
    mLL = []
    for mX17 in massX17Scan:
        startingPs = np.array([0, 37500, 27500, 135000, 50000, mX17], dtype=float)
        values, errors, fval, valid = BB2DLLFiniteMC.getMaxLikelihood(hdata, hToyMC, binsdatax, binsdatay, startingPs, plotFigure = False, parametrizedX17 = parametrizeX17, FixMass = True)
        with open(workDir + f'{prefix}_globalPValue_SEED{SEED}.txt', 'a') as f:
            f.write(f'{SEED + i} {mX17} {fval}\n')
        mLL.append(fval)
    
    # Interpolate
    mLL = np.array(mLL)
    f = interp1d(massX17Scan, -mLL + fvalH0, kind='cubic')
    
    # Compute the number of crossings
    threshold = pvalueC0
    
    # How many times does the function go above threshold?
    x = np.linspace(massX17Scan[0], massX17Scan[-1], 1000)
    y = f(x)
    nCrossings = 0
    for I in range(len(y) - 1):
        if y[I] < threshold and y[I+1] > threshold:
            nCrossings += 1
    with open(workDir + f'{prefix}_upperCrossing_SEED{SEED}.txt', 'a') as f:
        f.write(f'{SEED + i} {nCrossings}\n')
    plt.figure(figsize=(28, 14), dpi=100)
    plt.title('Profile likelihood - upcrossings')
    plt.plot(x, y, linewidth=5, color = cm.coolwarm(0.), label='Number of upcrossings: ' + str(nCrossings))
    plt.plot(massX17Scan, -mLL + fvalH0, 'o', linewidth=5, markersize=20, color = cm.coolwarm(0.))
    plt.plot(x, np.ones(len(x))*threshold, linewidth=5, color = 'black', label='c0')
    plt.legend()
    plt.ylabel('q(m)')
    plt.xlabel('X17 mass [MeV/c$^2$]')
    plt.grid()
    plt.savefig(workDir + f'{prefix}_profileLikelihood_SEED{SEED}_toy{i}.png', bbox_inches='tight')
    
    print('Global p-value elapsed time: ', time() - startTime)
    return


if __name__ == '__main__':
    #args, unknown = argparser()
    args = argparser()
    SEED = args.seed
    workDir = args.workDir
    reset = args.reset
    referenceFile = args.referenceFile
    dataF = args.dataFile
    prefix = args.prefix
    varyReference = args.varyReference
    ToySample = args.ToySample
    numberToys = args.numberToys
    nX17Toy = args.numberX17
    massX17Toy = args.massX17
    posterioriFC = args.posterioriFC
    parametrizeX17 = args.parametrizeX17
    plotToy = args.plotToy
    resetFC = args.resetFC
    globalPValue = args.globalPValue
    pvalueMin = args.pvalueMin
    pvalueMax = args.pvalueMax
    pvalueC0 = args.pvalueC0
    numberPL = args.numberPL
    ecodeCoding = args.ecodeCoding
    sampleHistograms = args.sampleHistograms
    BKGonly = args.BKGonly
    
    # Test Toy
    #ToySample = True
    #SEED = 0
    #numberToys = 2
    #nX17Toy = 391.5
    #nX17Toy = 0
    #nX17Toy = 2000
    #massX17Toy = 16.825024277990128
    #massX17Toy = 16.
    #massX17Toy = 18.
    #parametrizeX17 = True
    #plotToy = True
    #posterioriFC = False
    #dataF = 'X17MC2021'
    #referenceFile = 'X17referenceRealistic.root'
    #referenceFile = 'X17reference.root'
    #prefix = 'CurrentStatistics'
    #prefix = 'IdealStatistics'
    #numberPL = 15
    #resetFC = True
    
    ## Test global p-value
    #globalPValue = True
    #pvalueMin = 15
    #pvalueMax = 18.15
    #pvalueC0 = 1
    #parametrizeX17 = True
    #dataF = 'X17MC2021_s0'
    #numberToys = 100
    #SEED = 0
    
    if args.profileLikelihood:
        for I in range(numberToys):
            sampleMass(_Nbkg = 250000, _fIPC18 = 0.20, _fIPC15 = 0.11, _fEPC18 = 0.54, _Nx17 = nX17Toy, year = 2021, SEED = SEED + I, workDir = workDir, fileName=dataF)
            print(args.numberPL)
            
            startTime = time()
            dataFile = f'{dataF}_s{SEED + I}.root'
            MCFile = f'{referenceFile}'
            hMX, binsXMCx, binsXMCy = BB2DLLFiniteMC.loadMC(MCFile, workDir = workDir)
            hdata, binsdatax, binsdatay = BB2DLLFiniteMC.loadData(dataFile, workDir = workDir)
            
            startTime = time()
            startingPs = np.array([450, 37500, 27500, 135000, 50000, 17], dtype=float)
            values, errors, fval, valid = BB2DLLFiniteMC.getMaxLikelihood(hdata, hMX, binsdatax, binsdatay, startingPs, plotFigure = False, parametrizedX17 = parametrizeX17)
            bestNX17 = values[0]
            bestMass = 17
            if parametrizeX17:
                bestMass = values[5]
            fBest = fval
            
            # Create grid to scan profile likelihood
            nX17Scan = np.linspace(args.nX17plMin, args.nX17plMax, args.numberPL)
            
            massX17Scan = []
            
            pScan = []
            if parametrizeX17:
                MIN = args.mX17plMin
                MAX = args.mX17plMax
                massX17Scan = np.linspace(MIN, MAX, args.numberPL)
            
            print(nX17Scan, massX17Scan)
            if args.profileLikelihood2D and len(massX17Scan) > 1:
                # Create file
                with open(workDir + f'{prefix}_profileLikelihood_SEED{SEED + I}.txt', 'w') as f:
                    f.write('#nX17 fval\n')
                    f.write(f'{bestNX17} {bestMass} {fBest} {fBest}\n')
                for i in range(len(nX17Scan)):
                    for j in range(len(massX17Scan)):
                        startingPs = np.array([nX17Scan[i], 37500, 27500, 135000, 50000, massX17Scan[j]], dtype=float)
                        values, errors, fval, valid = BB2DLLFiniteMC.getMaxLikelihood(hdata, hMX, binsdatax, binsdatay, startingPs, plotFigure = False, doNullHyphotesis=True, parametrizedX17 = parametrizeX17)

                        pScan.append(fval)
                        # append to file
                        with open(workDir + f'{prefix}_profileLikelihood_SEED{SEED + I}.txt', 'a') as f:
                            f.write(f'{nX17Scan[i]} {massX17Scan[j]} {fval} {fBest}\n')
                print('Profile likelihood elapsed time: ', time() - startTime)
            else:
                # Create file
                with open(workDir + f'{prefix}_profileLikelihood_SEED{SEED + I}.txt', 'w') as f:
                    f.write('#nX17 fval\n')
                    f.write(f'{bestNX17} {fBest}\n')
                for i in range(len(nX17Scan)):
                    if nX17Scan[i] != bestNX17:
                        startingPs = np.array([nX17Scan[i], 37500, 27500, 135000, 50000, bestMass], dtype=float)
                        values, errors, fval, valid = BB2DLLFiniteMC.getMaxLikelihood(hdata, hMX, binsdatax, binsdatay, startingPs, plotFigure = False, doNullHyphotesis=True, parametrizedX17 = parametrizeX17)

                        pScan.append(fval)
                        # append to file
                        with open(workDir + f'{prefix}_profileLikelihood_SEED{SEED + I}.txt', 'a') as f:
                            f.write(f'{nX17Scan[i]} {fval} {fBest}\n')
                    
                
                if len(massX17Scan) > 1:
                    with open(workDir + f'{prefix}_profileLikelihood_SEED{SEED + I}.txt', 'a') as f:
                        f.write('#mX17 fval\n')
                        f.write(f'{bestMass} {fBest} {fBest}\n')
                    for i in range(len(massX17Scan)):
                        if massX17Scan[i] != bestMass:
                            startingPs = np.array([bestNX17, 37500, 27500, 135000, 50000, massX17Scan[i]], dtype=float)
                            values, errors, fval, valid = BB2DLLFiniteMC.getMaxLikelihood(hdata, hMX, binsdatax, binsdatay, startingPs, plotFigure = False, doNullHyphotesis=True, parametrizedX17 = parametrizeX17)

                            pScan.append(fval)
                            with open(workDir + f'{prefix}_profileLikelihood_SEED{SEED + I}.txt', 'a') as f:
                                f.write(f'{massX17Scan[i]} {fval} {fBest}\n')
            print('Profile likelihood elapsed time: ', time() - startTime)
    
    elif globalPValue and parametrizeX17:
        # Profile the likelihood while scanning the mass
        MIN = pvalueMin
        MAX = pvalueMax
        if MIN < 15:
            MIN = 15
        if MAX > 18.15:
            MAX = 18.15
            massX17Scan = np.linspace(MIN, MAX, numberPL)
        startTime = time()
        
        with open(workDir + f'{prefix}_globalPValue_SEED{SEED}.txt', 'w') as f:
            f.write('#SEED mX17 fval\n')
        
        with open(workDir + f'{prefix}_upperCrossing_SEED{SEED}.txt', 'w') as f:
            f.write('#SEED nCrossings\n')
        
        for i in range(numberToys):
            globalPvalue(i, workDir, referenceFile, nX17Toy, parametrizeX17, pvalueC0, MIN, MAX, prefix, startTime, numberPL)
    
    elif ToySample:
        dataFile = f'{dataF}.root'
        MCFile = f'{referenceFile}'
        hMX, binsXMCx, binsXMCy = BB2DLLFiniteMC.loadMC(MCFile, workDir = workDir)
        hdata, binsdatax, binsdatay = BB2DLLFiniteMC.loadData(dataFile, workDir = workDir)
        
        X, Y = np.meshgrid((binsdatax[:-1] + binsdatax[1:])/2, (binsdatay[:-1] + binsdatay[1:])/2)
        print(massX17Toy)
        hMX[0] = BB2DLLFiniteMC.nMCXtotParametrized*SigLikX17.AngleVSEnergySum(X, Y, massX17Toy, BB2DLLFiniteMC.dthMin, BB2DLLFiniteMC.dthMax, BB2DLLFiniteMC.dthnBins, BB2DLLFiniteMC.esumMin, BB2DLLFiniteMC.esumMax, BB2DLLFiniteMC.esumnBins, dthRes = 9.5, esumRes = 1.15)
        startingPs = np.array([450, 37500, 27500, 135000, 50000, 17], dtype=float)
        if ecodeCoding == 1:
            startingPs = np.array([450, 106889.88591294855, 0.4154335251499399, 11627.154551231886, 0.00029258110335583787, 74879.0672728438, 0.005317345671056906, 150686.2621480443, 157176.2036992142], dtype=float)
        
        if posterioriFC:
            values, errors, fval, valid = BB2DLLFiniteMC.getMaxLikelihood(hdata, hMX, binsdatax, binsdatay, startingPs, plotFigure = False, parametrizedX17 = parametrizeX17)
            # Get Aji
            startingPs = values
            nMCXtot = hMX[0].sum()
            nMCXe15 = hMX[1].sum()
            nMCXi15 = hMX[2].sum()
            nMCXe18 = hMX[3].sum()
            nMCXi18 = hMX[4].sum()
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
            if parametrizeX17:
                nMCXtot = BB2DLLFiniteMC.nMCXtotParametrized
            if ecodeCoding == 0:
                pvalues = values[:5]/np.array([nMCXtot, nMCXe15, nMCXi15, nMCXe18, nMCXi18])
                if parametrizeX17:
                    val, AIJ, TI = BB2DLLFiniteMC.LogLikelihood([pvalues[0], pvalues[1], pvalues[2], pvalues[3], pvalues[4]], hdata, hMX, True, Kstart=1)
                else:
                    val, AIJ, TI = BB2DLLFiniteMC.LogLikelihood([pvalues[0], pvalues[1], pvalues[2], pvalues[3], pvalues[4]], hdata, hMX, True)
                
            elif ecodeCoding == 1:
                pvalues = values[:9]/np.array([nMCXtot, nMCXi176, nMCXi179, nMCXi181, nMCXi146, nMCXi149, nMCXi151, nMCXe18, nMCXe15])
                if parametrizeX17:
                    val, AIJ, TI = BB2DLLFiniteMC.LogLikelihood([pvalues[0], pvalues[1], pvalues[2], pvalues[3], pvalues[4], pvalues[5], pvalues[6], pvalues[7], pvalues[8]], hdata, hMX, True, Kstart=1, ecodeCoding = ecodeCoding)
                else:
                    val, AIJ, TI = BB2DLLFiniteMC.LogLikelihood([pvalues[0], pvalues[1], pvalues[2], pvalues[3], pvalues[4], pvalues[5], pvalues[6], pvalues[7], pvalues[8]], hdata, hMX, True, ecodeCoding = ecodeCoding)
            
            hBestFit = []
            for j in range(len(hMX)):
                hj = []
                for I in range(BB2DLLFiniteMC.dthnBins):
                    htemp = []
                    for J in range(BB2DLLFiniteMC.esumnBins):
                        htemp.append(AIJ[I*BB2DLLFiniteMC.esumnBins + J][j])
                    
                    hj.append(np.array(htemp))
                    
                hBestFit.append(np.array(hj))

            # Reshape
            hBestFit = np.array(hBestFit)
            hMX = hBestFit
        
        startingPs[0] = nX17Toy
        startingPs[massIndex] = massX17Toy
        
        if ecodeCoding == 0:
            nMCXtot = hMX[0].sum()
            nMCXe15 = hMX[1].sum()
            nMCXi15 = hMX[2].sum()
            nMCXe18 = hMX[3].sum()
            nMCXi18 = hMX[4].sum()
            p = np.array([nX17Toy/nMCXtot, startingPs[1]/nMCXe15, startingPs[2]/nMCXi15, startingPs[3]/nMCXe18, startingPs[4]/nMCXi18])
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
            p = np.array([nX17Toy/nMCXtot, startingPs[1]/nMCXi176, startingPs[2]/nMCXi179, startingPs[3]/nMCXi181, startingPs[4]/nMCXi146, startingPs[5]/nMCXi149, startingPs[6]/nMCXi151, startingPs[7]/nMCXe18, startingPs[8]/nMCXe15])
        
        # Check if file to store laratio of Toys exists
        my_file = Path(workDir + f'../fcAnalysis/{prefix}_lratio_SEED{SEED}_nX17{nX17Toy}_mX17{massX17Toy}.txt')
        if resetFC or not my_file.is_file():
            # Create file to store lratio of Toys
            with open(workDir + f'../fcAnalysis/{prefix}_lratio_SEED{SEED}_nX17{nX17Toy}_mX17{massX17Toy}.txt', 'w') as f:
                f.write('#lratio\n')
        
        startTime = time()
        # Do Data point
        if SEED == 0:
            print(startingPs)
            values, errors, fval, validH0 = BB2DLLFiniteMC.getMaxLikelihood(hdata, hMX, binsdatax, binsdatay, startingPs, plotFigure = plotToy, parametrizedX17 = parametrizeX17, doNullHyphotesis = True, ecodeCoding = ecodeCoding)
            lratio = fval
            values[0] = 450
            values[massIndex] = 17
            values, errors, fval, validH1 = BB2DLLFiniteMC.getMaxLikelihood(hdata, hMX, binsdatax, binsdatay, values, plotFigure = plotToy, parametrizedX17 = parametrizeX17, ecodeCoding = ecodeCoding)
            lratio -= fval
            
            # Store nX17Toy, massX17Toy, validH1, validH0, SEED, lratio
            with open(workDir + f'../fcAnalysis/{prefix}_lratio_SEED{SEED}_nX17{nX17Toy}_mX17{massX17Toy}.txt', 'a') as f:
                f.write(f'{nX17Toy} {massX17Toy} {validH1} {validH0} {True} {SEED} {lratio}\n')
        
        for i in range(numberToys):
            hMX[0] = BB2DLLFiniteMC.nMCXtotParametrized*SigLikX17.AngleVSEnergySum(X, Y, massX17Toy, BB2DLLFiniteMC.dthMin, BB2DLLFiniteMC.dthMax, BB2DLLFiniteMC.dthnBins, BB2DLLFiniteMC.esumMin, BB2DLLFiniteMC.esumMax, BB2DLLFiniteMC.esumnBins, dthRes = 9.5, esumRes = 1.15)
            # Sample data ToyMC
            htemp = np.zeros((BB2DLLFiniteMC.dthnBins, BB2DLLFiniteMC.esumnBins))
            for k in range(len(hMX)):
                htemp += hMX[k]*p[k]
            hToyData = BB2DLLFiniteMC.sampleToyMC(htemp, SEED + i)
            
            # Sample reference ToyMCs
            hToyMC = []
            for j in range(len(hMX)):
                # Do not sample X17 if parametrized
                if j == 0 and parametrizeX17:
                    hToyMC.append(hMX[j])
                else:
                    hToyMC.append(BB2DLLFiniteMC.sampleToyMC(hMX[j], SEED + i))
            values, errors, fval, validH0 = BB2DLLFiniteMC.getMaxLikelihood(hToyData, hToyMC, binsdatax, binsdatay, startingPs, plotFigure = plotToy, parametrizedX17 = parametrizeX17, doNullHyphotesis = True)
            lratio = fval
            values[0] = nX17Toy
            if parametrizeX17:
                values[massIndex] = massX17Toy
            values, errors, fval, validH1 = BB2DLLFiniteMC.getMaxLikelihood(hToyData, hToyMC, binsdatax, binsdatay, values, plotFigure = plotToy, parametrizedX17 = parametrizeX17)
            lratio -= fval
            
            # Store nX17Toy, massX17Toy, validH1, validH0, SEED, lratio
            with open(workDir + f'../fcAnalysis/{prefix}_lratio_SEED{SEED}_nX17{nX17Toy}_mX17{massX17Toy}.txt', 'a') as f:
                f.write(f'{nX17Toy} {massX17Toy} {validH1} {validH0} {False} {SEED+i} {lratio}\n')
            
            print('Toy MC elapsed time: ', time() - startTime)
    
    else:
        startTime = time()
        # Fit data to get Ajis as template is posteriorFC and sampleHistograms are True
        if sampleHistograms:
            dataFile = f'{dataF}.root'
            MCFile = f'{referenceFile}'
            hMX, binsXMCx, binsXMCy = BB2DLLFiniteMC.loadMC(MCFile, workDir = workDir, ecodeCoding=ecodeCoding)
            hdata, binsdatax, binsdatay = BB2DLLFiniteMC.loadData(dataFile, workDir = workDir)
            
            startingPs = np.array([450, 37500, 27500, 135000, 50000, 17], dtype=float)
            if ecodeCoding == 1:
                startingPs = np.array([450, 106889.88591294855, 0.4154335251499399, 11627.154551231886, 0.00029258110335583787, 74879.0672728438, 0.005317345671056906, 150686.2621480443, 157176.2036992142, 17], dtype=float)
            if posterioriFC:
                values, errors, fval, valid = BB2DLLFiniteMC.getMaxLikelihood(hdata, hMX, binsdatax, binsdatay, startingPs, plotFigure = plotToy, parametrizedX17 = parametrizeX17, ecodeCoding=ecodeCoding, fractionsActive = True, fractions = [0.7368421052631579, 0.4444444444444445, 0.4117647058823529]) 
                # Get Aji
                startingPs = values
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
                if parametrizeX17:
                    nMCXtot = BB2DLLFiniteMC.nMCXtotParametrized
                if ecodeCoding == 0:
                    pvalues = values[:5]/np.array([nMCXtot, nMCXe15, nMCXi15, nMCXe18, nMCXi18])
                    if parametrizeX17:
                        val, AIJ, TI = BB2DLLFiniteMC.LogLikelihood([pvalues[0], pvalues[1], pvalues[2], pvalues[3], pvalues[4]], hdata, hMX, True, Kstart=1)
                    else:
                        val, AIJ, TI = BB2DLLFiniteMC.LogLikelihood([pvalues[0], pvalues[1], pvalues[2], pvalues[3], pvalues[4]], hdata, hMX, True)
                    
                elif ecodeCoding == 1:
                    pvalues = values[:9]/np.array([nMCXtot, nMCXi176, nMCXi179, nMCXi181, nMCXi146, nMCXi149, nMCXi151, nMCXe18, nMCXe15])
                    if parametrizeX17:
                        val, AIJ, TI = BB2DLLFiniteMC.LogLikelihood([pvalues[0], pvalues[1], pvalues[2], pvalues[3], pvalues[4], pvalues[5], pvalues[6], pvalues[7], pvalues[8]], hdata, hMX, True, Kstart=1)
                    else:
                        val, AIJ, TI = BB2DLLFiniteMC.LogLikelihood([pvalues[0], pvalues[1], pvalues[2], pvalues[3], pvalues[4], pvalues[5], pvalues[6], pvalues[7], pvalues[8]], hdata, hMX, True)
                
                hBestFit = []
                for j in range(len(hMX)):
                    hj = []
                    for I in range(BB2DLLFiniteMC.dthnBins):
                        htemp = []
                        for J in range(BB2DLLFiniteMC.esumnBins):
                            htemp.append(AIJ[I*BB2DLLFiniteMC.esumnBins + J][j])
                        
                        hj.append(np.array(htemp))
                        
                    hBestFit.append(np.array(hj))

                # Reshape
                hBestFit = np.array(hBestFit)
                hMX = hBestFit
                hdataOriginal = hdata
                hMXOriginal = hMX
            
        for i in range(args.nSamples):
            # Here, the sample2D function is used
            if not sampleHistograms:
                if args.reuseMC:
                    my_file = Path(workDir + f'{dataF}_s{SEED + i}.root')
                    if not my_file.is_file():
                        sampleMass(_Nbkg = 250000, _fIPC18 = 0.20, _fIPC15 = 0.11, _fEPC18 = 0.54, _Nx17 = nX17Toy, year = 2021, SEED = SEED + i, workDir = workDir, fileName=dataF)
                    if varyReference:
                        if referenceFile.find('Realistic') < 0:
                            my_file = Path(workDir + f'{referenceFile}_s{SEED + i + 299792458}.root')
                            if not my_file.is_file():
                                sampleMass(_Nbkg = 400000, _fIPC18 = 0.25, _fIPC15 = 0.25, _fEPC18 = 0.25, _Nx17 = 100000, year = 2021, SEED = SEED + i + 299792458, workDir = workDir, fileName = referenceFile)
                        else:
                            my_file = Path(workDir + f'{referenceFile}_s{SEED + i + 662607015}.root')
                            if not my_file.is_file():
                                sampleMass(_Nbkg = 220000, _fIPC18 = 0.45454545454545453, _fIPC15 = 0.45454545454545453, _fEPC18 = 0.045454545454545453, _Nx17 = 100000, year = 2021, SEED = SEED + i + 662607015, workDir = workDir, fileName = referenceFile)
                else:
                    sampleMass(_Nbkg = 250000, _fIPC18 = 0.20, _fIPC15 = 0.11, _fEPC18 = 0.54, _Nx17 = nX17Toy, year = 2021, SEED = SEED + i, workDir = workDir, fileName=dataF)
                    if varyReference:
                        if referenceFile.find('Realistic') < 0:
                            sampleMass(_Nbkg = 400000, _fIPC18 = 0.25, _fIPC15 = 0.25, _fEPC18 = 0.25, _Nx17 = 100000, year = 2021, SEED = SEED + i + 299792458, workDir = workDir, fileName = referenceFile)
                        else:
                            sampleMass(_Nbkg = 220000, _fIPC18 = 0.45454545454545453, _fIPC15 = 0.45454545454545453, _fEPC18 = 0.045454545454545453, _Nx17 = 100000, year = 2021, SEED = SEED + i + 662607015, workDir = workDir, fileName = referenceFile)
                
                
                dataFile = f'{dataF}_s{SEED + i}.root'
                MCFile = f'{referenceFile}'
                if varyReference:   
                    if referenceFile.find('Realistic') < 0:
                        MCFile = f'{referenceFile}_s{SEED + i + 299792458}.root'
                    else:
                        MCFile = f'{referenceFile}_s{SEED + i + 662607015}.root'
                print(MCFile)
                hMX, binsXMCx, binsXMCy = BB2DLLFiniteMC.loadMC(MCFile, workDir = workDir)
                hdata, binsdatax, binsdatay = BB2DLLFiniteMC.loadData(dataFile, workDir = workDir)
                startingPs = np.array([nX17Toy, 37500, 27500, 135000, 50000, 17], dtype = float)
            # Here the templates and toys are sampled from the original template or from the fitted templates
            else:
                if ecodeCoding == 0:
                    if parametrizeX17:
                        nMCXtot = BB2DLLFiniteMC.nMCXtotParametrized
                    else:
                        nMCXtot = hMXOriginal[0].sum()
                    nMCXe15 = hMXOriginal[1].sum()
                    nMCXi15 = hMXOriginal[2].sum()
                    nMCXe18 = hMXOriginal[3].sum()
                    nMCXi18 = hMXOriginal[4].sum()
                    p = np.array([nX17Toy/nMCXtot, startingPs[1]/nMCXe15, startingPs[2]/nMCXi15, startingPs[3]/nMCXe18, startingPs[4]/nMCXi18])
                elif ecodeCoding == 1:
                    nMCXtot = 1e5
                    nMCXi176 = hMXOriginal[1].sum()
                    nMCXi179 = hMXOriginal[2].sum()
                    nMCXi181 = hMXOriginal[3].sum()
                    nMCXi146 = hMXOriginal[4].sum()
                    nMCXi149 = hMXOriginal[5].sum()
                    nMCXi151 = hMXOriginal[6].sum()
                    nMCXe18 = hMXOriginal[7].sum()
                    nMCXe15 = hMXOriginal[8].sum()
                    p = np.array([nX17Toy/nMCXtot, startingPs[1]/nMCXi176, startingPs[2]/nMCXi179, startingPs[3]/nMCXi181, startingPs[4]/nMCXi146, startingPs[5]/nMCXi149, startingPs[6]/nMCXi151, startingPs[7]/nMCXe18, startingPs[8]/nMCXe15])
                # Sample data ToyMC
                htemp = np.zeros((BB2DLLFiniteMC.dthnBins, BB2DLLFiniteMC.esumnBins))
                for k in range(len(hMXOriginal)):
                    htemp += hMXOriginal[k]*p[k]
                hdata = BB2DLLFiniteMC.sampleToyMC(hdataOriginal, SEED + i)
                
                # Sample reference ToyMCs
                hMX = []
                for j in range(len(hMXOriginal)):
                    # Do not sample X17 if parametrized
                    if j == 0 and parametrizeX17:
                        hMX.append(hMXOriginal[j])
                    else:
                        hMX.append(BB2DLLFiniteMC.sampleToyMC(hMXOriginal[j], SEED + i))
            
            if not BKGonly:
                values, errors, fval, valid = BB2DLLFiniteMC.getMaxLikelihood(hdata, hMX, binsdatax, binsdatay, startingPs, plotFigure = plotToy, parametrizedX17 = parametrizeX17, ecodeCoding=ecodeCoding, fractionsActive = True, fractions = [0.7368421052631579, 0.4444444444444445, 0.4117647058823529]) 
                execTime = time() - startTime
                
                startingPs[0] = 0
            valuesH0, errorsH0, fvalH0, validH0 = BB2DLLFiniteMC.getMaxLikelihood(hdata, hMX, binsdatax, binsdatay, startingPs, plotFigure = plotToy, doNullHyphotesis=True, parametrizedX17 = parametrizeX17, ecodeCoding=ecodeCoding, fractionsActive = True, fractions = [0.7368421052631579, 0.4444444444444445, 0.4117647058823529]) 
            if BKGonly:
                values = valuesH0
                errors = errorsH0
                fval = fvalH0
                valid = validH0
                execTime = time() - startTime
            
            DOF = 1
            if parametrizeX17:
                DOF = 2
            lratio, pvalue, sigma = BB2DLLFiniteMC.computeSignificance(fvalH0, fval, DOF)
            
            execTime2 = time() - execTime - startTime
            
            # Append results to file
            if parametrizeX17:
                if i == 0 and reset:
                    if ecodeCoding == 0:
                        with open(workDir + f'{prefix}_results_SEED{SEED}.txt', 'w') as f:
                            f.write('#nSig nSigErr nEpc15 nEpc15Err nIpc15 nIpc15Err nEpc18 nEpc18Err nIpc18 nIpc18Err fval valid fvalH0 validH0 lratio pvalue sigma ExecTime ExecTimeH0 mX17\n')
                            f.write(f'{values[0]} {errors[0]} {values[1]} {errors[1]} {values[2]} {errors[2]} {values[3]} {errors[3]} {values[4]} {errors[4]} {fval} {valid} {fvalH0} {validH0} {lratio} {pvalue} {sigma} {execTime} {execTime2} {values[5]}\n')
                    elif ecodeCoding == 1:
                        with open(workDir + f'{prefix}_results_SEED{SEED}.txt', 'w') as f:
                            f.write('#nSig nSigErr nI176 nI176Err nI179 nI179Err nI181 nI181Err nI146 nI146Err nI149 nI149Err nI151 nI151Err nEpc18 nEpc18Err nEpc15 nEpc15Err fval valid fvalH0 validH0 lratio pvalue sigma ExecTime ExecTimeH0 mX17\n')
                            f.write(f'{values[0]} {errors[0]} {values[1]} {errors[1]} {values[2]} {errors[2]} {values[3]} {errors[3]} {values[4]} {errors[4]} {values[5]} {errors[5]} {values[6]} {errors[6]} {values[7]} {errors[7]} {values[8]} {errors[8]} {fval} {valid} {fvalH0} {validH0} {lratio} {pvalue} {sigma} {execTime} {execTime2} {values[9]}\n')
                else:
                    if ecodeCoding == 0:
                        with open(workDir + f'{prefix}_results_SEED{SEED}.txt', 'a') as f:
                            f.write(f'{values[0]} {errors[0]} {values[1]} {errors[1]} {values[2]} {errors[2]} {values[3]} {errors[3]} {values[4]} {errors[4]} {fval} {valid} {fvalH0} {validH0} {lratio} {pvalue} {sigma} {execTime} {execTime2} {values[5]}\n')
                    elif ecodeCoding == 1:
                        with open(workDir + f'{prefix}_results_SEED{SEED}.txt', 'a') as f:
                            f.write(f'{values[0]} {errors[0]} {values[1]} {errors[1]} {values[2]} {errors[2]} {values[3]} {errors[3]} {values[4]} {errors[4]} {values[5]} {errors[5]} {values[6]} {errors[6]} {values[7]} {errors[7]} {values[8]} {errors[8]} {fval} {valid} {fvalH0} {validH0} {lratio} {pvalue} {sigma} {execTime} {execTime2} {values[9]}\n')
            else:
                if i == 0 and reset:
                    if ecodeCoding == 0:
                        with open(workDir + f'{prefix}_results_SEED{SEED}.txt', 'w') as f:
                            f.write('#nSig nSigErr nEpc15 nEpc15Err nIpc15 nIpc15Err nEpc18 nEpc18Err nIpc18 nIpc18Err fval valid fvalH0 validH0 lratio pvalue sigma ExecTime ExecTimeH0\n')
                            f.write(f'{values[0]} {errors[0]} {values[1]} {errors[1]} {values[2]} {errors[2]} {values[3]} {errors[3]} {values[4]} {errors[4]} {fval} {valid} {fvalH0} {validH0} {lratio} {pvalue} {sigma} {execTime} {execTime2}\n')
                    elif ecodeCoding == 1:
                        with open(workDir + f'{prefix}_results_SEED{SEED}.txt', 'w') as f:
                            f.write('#nSig nSigErr nI176 nI176Err nI179 nI179Err nI181 nI181Err nI146 nI146Err nI149 nI149Err nI151 nI151Err nEpc18 nEpc18Err nEpc15 nEpc15Err fval valid fvalH0 validH0 lratio pvalue sigma ExecTime ExecTimeH0\n')
                            f.write(f'{values[0]} {errors[0]} {values[1]} {errors[1]} {values[2]} {errors[2]} {values[3]} {errors[3]} {values[4]} {errors[4]} {values[5]} {errors[5]} {values[6]} {errors[6]} {values[7]} {errors[7]} {values[8]} {errors[8]} {fval} {valid} {fvalH0} {validH0} {lratio} {pvalue} {sigma} {execTime} {execTime2}\n')
                else:
                    if ecodeCoding == 0:
                        with open(workDir + f'{prefix}_results_SEED{SEED}.txt', 'a') as f:
                            f.write(f'{values[0]} {errors[0]} {values[1]} {errors[1]} {values[2]} {errors[2]} {values[3]} {errors[3]} {values[4]} {errors[4]} {fval} {valid} {fvalH0} {validH0} {lratio} {pvalue} {sigma} {execTime} {execTime2}\n')
                    elif ecodeCoding == 1:
                        with open(workDir + f'{prefix}_results_SEED{SEED}.txt', 'a') as f:
                            f.write(f'{values[0]} {errors[0]} {values[1]} {errors[1]} {values[2]} {errors[2]} {values[3]} {errors[3]} {values[4]} {errors[4]} {values[5]} {errors[5]} {values[6]} {errors[6]} {values[7]} {errors[7]} {values[8]} {errors[8]} {fval} {valid} {fvalH0} {validH0} {lratio} {pvalue} {sigma} {execTime} {execTime2}\n')
                    
            
        
        
