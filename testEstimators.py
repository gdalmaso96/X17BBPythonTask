# Test estimators from BB2DLLFiniteMC.py
from sample2D import sampleMass
import numpy as np
import BB2DLLFiniteMC
# add argument parsin
import argparse
from time import time
from pathlib import Path
import SigLikX17

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
    parser.add_argument('-ts', '--ToySample', type=bool, default=False, help='Produced ToyMC sample')
    parser.add_argument('-nt', '--numberToys', type=int, default=1000, help='Number of toys to produce')
    parser.add_argument('-nX17', '--numberX17', type=int, default=450, help='Number of X17 events to produce in ToyMC')
    parser.add_argument('-mX17', '--massX17', type=int, default=17, help='Mass of X17 events to produce in ToyMC')
    parser.add_argument('-pFC', '--posterioriFC', type=bool, default=False, help='Use a posteriori FC')
    parser.add_argument('-rFC', '--resetFC', type=bool, default=False, help='Replace FC file')
    parser.add_argument('-st', '--saveToy', type=bool, default=False, help='Save ToyMC')
    parser.add_argument('-pT', '--plotToy', type=bool, default=False, help='Plot ToyMC')
    return parser.parse_args()


if __name__ == '__main__':
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
    
    
    if ToySample:
        dataFile = f'{dataF}.root'
        MCFile = f'{referenceFile}'
        hMX, binsXMCx, binsXMCy = BB2DLLFiniteMC.loadMC(MCFile, workDir = workDir)
        
        binsdatax = np.linspace(BB2DLLFiniteMC.dthMin, BB2DLLFiniteMC.dthMax, BB2DLLFiniteMC.dthnBins + 1)
        binsdatay = np.linspace(BB2DLLFiniteMC.esumMin, BB2DLLFiniteMC.esumMax, BB2DLLFiniteMC.esumnBins + 1)
        X, Y = np.meshgrid((binsdatax[:-1] + binsdatax[1:])/2, (binsdatay[:-1] + binsdatay[1:])/2)
        hMX[0] = BB2DLLFiniteMC.nMCXtotParametrized*SigLikX17.AngleVSEnergySum(X, Y, massX17Toy, BB2DLLFiniteMC.dthMin, BB2DLLFiniteMC.dthMax, BB2DLLFiniteMC.dthnBins, BB2DLLFiniteMC.esumMin, BB2DLLFiniteMC.esumMax, BB2DLLFiniteMC.esumnBins, dthRes = 9.5, esumRes = 1.15)
        startingPs = np.array([450, 37500, 27500, 135000, 50000, 17])
        
        if posterioriFC:
            hdata, binsdatax, binsdatay = BB2DLLFiniteMC.loadData(dataFile, workDir = workDir)
            values, errors, fval, accurate = BB2DLLFiniteMC.getMaxLikelihood(hdata, hMX, binsdatax, binsdatay, startingPs, plotFigure = True, parametrizedX17 = args.parametrizeX17)
            # Get Aji
            startingPs = values
        
        startingPs[0] = nX17Toy
        startingPs[5] = massX17Toy
        
        nMCXtot = hMX[0].sum()
        nMCXe15 = hMX[1].sum()
        nMCXi15 = hMX[2].sum()
        nMCXe18 = hMX[3].sum()
        nMCXi18 = hMX[4].sum()
        p = np.array([nX17Toy/nMCXtot, startingPs[1]/nMCXe15, startingPs[2]/nMCXi15, startingPs[3]/nMCXe18, startingPs[4]/nMCXi18])
        
        # Check if file to store laratio of Toys exists
        my_file = Path(workDir + f'../fcAnalysis/{prefix}_lratio_SEED{SEED}.txt')
        if args.resetFC or not my_file.is_file():
            # Create file to store lratio of Toys
            with open(workDir + f'../fcAnalysis/{prefix}_lratio_SEED{SEED}.txt', 'w') as f:
                f.write('#lratio\n')
        
        startTime = time()
        for i in range(numberToys):
            # Sample data ToyMC
            htemp = hMX[0]*p[0] + hMX[1]*p[1] + hMX[2]*p[2] + hMX[3]*p[3] + hMX[4]*p[4]
            hToyData = BB2DLLFiniteMC.sampleToyMC(htemp, SEED + i)
            
            # Sample reference ToyMCs
            hToyMC = []
            for j in range(len(hMX)):
                # Do not sample X17 if parametrized
                if j == 0 and args.parametrizeX17:
                    hToyMC.append(hMX[j])
                else:
                    hToyMC.append(BB2DLLFiniteMC.sampleToyMC(hMX[j], SEED + i))
            values, errors, fval, accurateH0 = BB2DLLFiniteMC.getMaxLikelihood(hToyData, hToyMC, binsdatax, binsdatay, startingPs, plotFigure = args.plotToy, parametrizedX17 = args.parametrizeX17, doNullHyphotesis = True)
            lratio = fval
            values[0] = nX17Toy
            values[5] = massX17Toy
            values, errors, fval, accurateH1 = BB2DLLFiniteMC.getMaxLikelihood(hToyData, hToyMC, binsdatax, binsdatay, values, plotFigure = args.plotToy, parametrizedX17 = args.parametrizeX17)
            lratio -= fval
            
            # Store nX17Toy, massX17Toy, accurateH1, accurateH0, SEED, lratio
            with open(workDir + f'../fcAnalysis/{prefix}_lratio_SEED{SEED}.txt', 'a') as f:
                f.write(f'{nX17Toy} {massX17Toy} {accurateH1} {accurateH0} {SEED+i} {lratio}\n')
            
            print('Toy MC elapsed time: ', time() - startTime)
    
    else:
        for i in range(args.nSamples):
            if args.reuseMC:
                my_file = Path(workDir + f'{dataF}_s{SEED + i}.root')
                if not my_file.is_file():
                    sampleMass(_Nbkg = 250000, _fIPC18 = 0.20, _fIPC15 = 0.11, _fEPC18 = 0.54, _Nx17 = 450, year = 2021, SEED = SEED + i, workDir = workDir)
                if varyReference:
                    if referenceFile.find('Realistic') < 0:
                        my_file = Path(workDir + f'{referenceFile}_s{SEED + i + 299792458}.root')
                        if not my_file.is_file():
                            sampleMass(_Nbkg = 400000, _fIPC18 = 0.25, _fIPC15 = 0.25, _fEPC18 = 0.25, _Nx17 = 100000, year = 2021, SEED = SEED + i + 299792458, workDir = workDir, fileName = referenceFile)
                    else:
                        my_file = Path(workDir + f'{referenceFile}_s{SEED + i + 662607015}.root')
                        if not my_file.is_file():
                            sampleMass(_Nbkg = 220000, _fIPC18 = 0.45454545454545453, _fIPC15 = 0.045454545454545453, _fEPC18 = 0.45454545454545453, _Nx17 = 100000, year = 2021, SEED = SEED + i + 662607015, workDir = workDir, fileName = referenceFile)
            else:
                sampleMass(_Nbkg = 250000, _fIPC18 = 0.20, _fIPC15 = 0.11, _fEPC18 = 0.54, _Nx17 = 450, year = 2021, SEED = SEED + i, workDir = workDir)
                if varyReference:
                    if referenceFile.find('Realistic') < 0:
                        sampleMass(_Nbkg = 400000, _fIPC18 = 0.25, _fIPC15 = 0.25, _fEPC18 = 0.25, _Nx17 = 100000, year = 2021, SEED = SEED + i + 299792458, workDir = workDir, fileName = referenceFile)
                    else:
                        sampleMass(_Nbkg = 220000, _fIPC18 = 0.45454545454545453, _fIPC15 = 0.045454545454545453, _fEPC18 = 0.45454545454545453, _Nx17 = 100000, year = 2021, SEED = SEED + i + 662607015, workDir = workDir, fileName = referenceFile)
            
            startTime = time()
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
            startingPs = np.array([450, 37500, 27500, 135000, 50000, 17])
            values, errors, fval, accurate = BB2DLLFiniteMC.getMaxLikelihood(hdata, hMX, binsdatax, binsdatay, startingPs, plotFigure = True, parametrizedX17 = args.parametrizeX17)
            execTime = time() - startTime
            startingPs = np.array([0, 37500, 27500, 135000, 50000, 17])
            valuesH0, errorsH0, fvalH0, accurateH0 = BB2DLLFiniteMC.getMaxLikelihood(hdata, hMX, binsdatax, binsdatay, startingPs, plotFigure = False, doNullHyphotesis=True, parametrizedX17 = args.parametrizeX17)
            
            DOF = 1
            if args.parametrizeX17:
                DOF = 2
            lratio, pvalue, sigma = BB2DLLFiniteMC.computeSignificance(fvalH0, fval, DOF)
            
            execTime2 = time() - execTime - startTime
            
            # Append results to file
            if args.parametrizeX17:
                if i == 0 and reset:
                    with open(workDir + f'{prefix}_results_SEED{SEED}.txt', 'w') as f:
                        f.write('#nSig nSigErr nEpc15 nEpc15Err nIpc15 nIpc15Err nEpc18 nEpc18Err nIpc18 nIpc18Err fval accurate fvalH0 accurateH0 lratio pvalue sigma ExecTime ExecTimeH0 mX17\n')
                        f.write(f'{values[0]} {errors[0]} {values[1]} {errors[1]} {values[2]} {errors[2]} {values[3]} {errors[3]} {values[4]} {errors[4]} {fval} {accurate} {fvalH0} {accurateH0} {lratio} {pvalue} {sigma} {execTime} {execTime2} {values[5]}\n')
                else:
                    with open(workDir + f'{prefix}_results_SEED{SEED}.txt', 'a') as f:
                        f.write(f'{values[0]} {errors[0]} {values[1]} {errors[1]} {values[2]} {errors[2]} {values[3]} {errors[3]} {values[4]} {errors[4]} {fval} {accurate} {fvalH0} {accurateH0} {lratio} {pvalue} {sigma} {execTime} {execTime2} {values[5]}\n')
            else:
                if i == 0 and reset:
                    with open(workDir + f'{prefix}_results_SEED{SEED}.txt', 'w') as f:
                        f.write('#nSig nSigErr nEpc15 nEpc15Err nIpc15 nIpc15Err nEpc18 nEpc18Err nIpc18 nIpc18Err fval accurate fvalH0 accurateH0 lratio pvalue sigma ExecTime ExecTimeH0\n')
                        f.write(f'{values[0]} {errors[0]} {values[1]} {errors[1]} {values[2]} {errors[2]} {values[3]} {errors[3]} {values[4]} {errors[4]} {fval} {accurate} {fvalH0} {accurateH0} {lratio} {pvalue} {sigma} {execTime} {execTime2}\n')
                else:
                    with open(workDir + f'{prefix}_results_SEED{SEED}.txt', 'a') as f:
                        f.write(f'{values[0]} {errors[0]} {values[1]} {errors[1]} {values[2]} {errors[2]} {values[3]} {errors[3]} {values[4]} {errors[4]} {fval} {accurate} {fvalH0} {accurateH0} {lratio} {pvalue} {sigma} {execTime} {execTime2}\n')
        
        
