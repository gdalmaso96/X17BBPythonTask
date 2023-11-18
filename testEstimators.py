# Test estimators from BB2DLLFiniteMC.py
from sample2D import sampleMass
import numpy as np
import BB2DLLFiniteMC
# add argument parsin
import argparse
from time import time
from pathlib import Path

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
        
        
