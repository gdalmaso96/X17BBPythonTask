import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import matplotlib
from scipy.stats import chi2, norm
import argparse

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--files', type=str, default='results/results_SEED*.txt', help='Path to results file')
    return parser.parse_args()

if __name__ == '__main__':
    # set font size
    matplotlib.rcParams.update({'font.size': 35})
    
    # Import all results files
    files = argparser().files
    results = glob(files)

    prefix = files[:files.find('_SEED')]
    prefix = prefix[prefix.rfind('/')+1:]
    
    # Define arrays to store results
    nSig = []
    nSigErr = []
    nEpc15 = []
    nEpc15Err = []
    nIpc15 = []
    nIpc15Err = []
    nEpc18 = []
    nEpc18Err = []
    nIpc18 = []
    nIpc18Err = []
    fval = []
    valid = []
    fvalH0 = []
    validH0 = []
    lratio = []
    pvalue = []
    sigma = []
    execTime = []
    execTimeH0 = []
    mX17 = []
    
    # Loop over all results files
    for result in results:
        with open(result, 'r') as f:
            for line in f:
                if line[0] == '#':
                    continue
                else:
                    # Extract values from line
                    values = line.split()
                    nSig.append(float(values[0]))
                    nSigErr.append(float(values[1]))
                    nEpc15.append(float(values[2]))
                    nEpc15Err.append(float(values[3]))
                    nIpc15.append(float(values[4]))
                    nIpc15Err.append(float(values[5]))
                    nEpc18.append(float(values[6]))
                    nEpc18Err.append(float(values[7]))
                    nIpc18.append(float(values[8]))
                    nIpc18Err.append(float(values[9]))
                    fval.append(float(values[10]))
                    valid.append(bool(values[11]))
                    fvalH0.append(float(values[12]))
                    validH0.append(bool(values[13]))
                    lratio.append(float(values[14]))
                    pvalue.append(float(values[15]))
                    sigma.append(float(values[16]))
                    execTime.append(float(values[17]))
                    execTimeH0.append(float(values[18]))
                    if len(values) > 19:
                        mX17.append(float(values[19]))
                
    # Convert to numpy arrays
    nSig = np.array(nSig)
    nSigErr = np.array(nSigErr)
    nEpc15 = np.array(nEpc15)
    nEpc15Err = np.array(nEpc15Err)
    nIpc15 = np.array(nIpc15)
    nIpc15Err = np.array(nIpc15Err)
    nEpc18 = np.array(nEpc18)
    nEpc18Err = np.array(nEpc18Err)
    nIpc18 = np.array(nIpc18)
    nIpc18Err = np.array(nIpc18Err)
    fval = np.array(fval)
    valid = np.array(valid)
    fvalH0 = np.array(fvalH0)
    validH0 = np.array(validH0)
    lratio = np.array(lratio)
    pvalue = np.array(pvalue)
    sigma = np.array(sigma)
    execTime = np.array(execTime)
    execTimeH0 = np.array(execTimeH0)
    mX17 = np.array(mX17)
    
    # Plot results
    plt.figure(figsize=(4*14, 2*14), dpi=100)
    plt.suptitle(f'{prefix}, {len(nSig)} tests')
    plt.subplot(2, 4, 1)
    plt.hist(nSig, bins=50, label=f'mean = {np.mean(nSig):.2f}, std = {np.std(nSig):.2f}')
    plt.xlabel('nSig')
    plt.legend()
    plt.grid()
    
    plt.subplot(2, 4, 2)
    plt.hist(nEpc15, bins=50, label=f'mean = {np.mean(nEpc15):.2f}, std = {np.std(nEpc15):.2f}')
    plt.xlabel('nEpc15')
    plt.legend()
    plt.grid()
    
    plt.subplot(2, 4, 3)
    plt.hist(nIpc15, bins=50, label=f'mean = {np.mean(nIpc15):.2f}, std = {np.std(nIpc15):.2f}')
    plt.xlabel('nIpc15')
    plt.legend()
    plt.grid()
    
    plt.subplot(2, 4, 4)
    plt.hist(nEpc18, bins=50, label=f'mean = {np.mean(nEpc18):.2f}, std = {np.std(nEpc18):.2f}')
    plt.xlabel('nEpc18')
    plt.legend()
    plt.grid()
    
    plt.subplot(2, 4, 5)
    plt.hist(nIpc18, bins=50, label=f'mean = {np.mean(nIpc18):.2f}, std = {np.std(nIpc18):.2f}')
    plt.xlabel('nIpc18')
    plt.legend()
    plt.grid()
    
    plt.subplot(2, 4, 6)
    plt.bar(['True', 'False'], [np.sum(valid), np.sum(valid == False)])
    plt.xlabel('valid')
    plt.yscale('log')
    plt.grid()
    
    plt.subplot(2, 4, 7)
    lr = fvalH0 - fval
    sigma = chi2.sf(lr, 1)
    sigma = norm.isf(sigma*0.5)
    plt.hist(sigma, bins=50, label=f'mean = {np.mean(sigma):.2f}, std = {np.std(sigma):.2f}')
    plt.xlabel('sigma')
    plt.legend()
    plt.grid()
    
    plt.subplot(2, 4, 8)
    if len(mX17) > 0:
        plt.hist(mX17, bins=50, label=f'mean = {np.mean(mX17):.2f}, std = {np.std(mX17):.2f}')
        plt.xlabel(r'$\mathrm{X17 mass [MeV/c^2]}$')
        plt.legend()
        plt.grid()
    else:
        plt.hist(execTime, bins=50, label=f'mean = {np.mean(execTime):.2f}, std = {np.std(execTime):.2f}')
        plt.xlabel('Execution time')
        plt.legend()
        plt.grid()
    
    plt.savefig(f'{prefix}.png', bbox_inches='tight')

    plt.figure(figsize=(28, 14), dpi=100)
    plt.suptitle(f'{prefix}, {len(nSig)} tests')
    plt.subplot(1, 2, 1)

    plt.hist(execTime, bins=50, label=f'mean = {np.mean(execTime):.2f}, std = {np.std(execTime):.2f}')
    plt.xlabel('Execution time')
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.hist(execTimeH0, bins=50, label=f'mean = {np.mean(execTimeH0):.2f}, std = {np.std(execTimeH0):.2f}')
    plt.xlabel('Execution time H0')
    plt.legend()
    plt.grid()

    plt.savefig(f'{prefix}_execTime.png', bbox_inches='tight')

