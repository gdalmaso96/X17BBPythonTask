import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import matplotlib
from matplotlib import cm
from scipy.stats import chi2, norm
import BB2DLLFiniteMC
import argparse

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--files', type=str, default='results/VaryReferencebins20x14IdealStatisticsParametrized_results_SEED*.txt', help='Path to results file')
    return parser.parse_args()

if __name__ == '__main__':
    # set font size
    matplotlib.rcParams.update({'font.size': 35})
    
    # Import all results files
    files = argparser().files
    #files = "results/VaryReferencebins20x14CurrentStatisticsParametrized_500_results_SEED*.txt"
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
    plt.suptitle(f'Binning {prefix[prefix.find("bins") + 4:prefix.find("bins") + 9]}, {prefix[prefix.find("bins") + 9:prefix.find("Statistics")]} statistics, ' + r'$\mathcal{N}_{\mathrm{Sig}}$' +  f' =  {prefix[prefix.find("_") + 1: prefix.find("_results")]}, {len(nSig)} toy MCs')
    plt.subplot(2, 4, 1)
    H = plt.hist(nSig/np.mean(nSig), bins=50, label=f'mean = {np.mean(nSig):.2f}, std = {np.std(nSig):.2f},\n relative std = {1e2*np.std(nSig)/np.mean(nSig):.2f} %', color=cm.coolwarm(0.))
    print(H)
    plt.stairs(H[0], H[1], color=cm.coolwarm(0.), linewidth=8)
    plt.xlabel(r'$\mathcal{N}_{\mathrm{Sig}}/\hat{\mathcal{N}}_{\mathrm{Sig}}$')
    plt.legend()
    plt.grid()
    
    plt.subplot(2, 4, 2)
    H = plt.hist(nEpc15/np.mean(nEpc15), bins=50, label=f'mean = {np.mean(nEpc15):.2f}, std = {np.std(nEpc15):.2f},\n relative std = {1e2*np.std(nEpc15)/np.mean(nEpc15):.2f} %', color=cm.coolwarm(0.))
    plt.stairs(H[0], H[1], color=cm.coolwarm(0.), linewidth=8)
    plt.xlabel(r'$\mathcal{N}_{\mathrm{Epc15}}/\hat{\mathcal{N}}_{\mathrm{Epc15}}$')
    plt.legend()
    plt.grid()
    
    plt.subplot(2, 4, 3)
    H = plt.hist(nIpc15/np.mean(nIpc15), bins=50, label=f'mean = {np.mean(nIpc15):.2f}, std = {np.std(nIpc15):.2f},\n relative std = {1e2*np.std(nIpc15)/np.mean(nIpc15):.2f} %', color=cm.coolwarm(0.))
    plt.stairs(H[0], H[1], color=cm.coolwarm(0.), linewidth=8)
    plt.xlabel(r'$\mathcal{N}_{\mathrm{Ipc15}}/\hat{\mathcal{N}}_{\mathrm{Ipc15}}$')
    plt.legend()
    plt.grid()
    
    plt.subplot(2, 4, 4)
    H = plt.hist(nEpc18/np.mean(nEpc18), bins=50, label=f'mean = {np.mean(nEpc18):.2f}, std = {np.std(nEpc18):.2f},\n relative std = {1e2*np.std(nEpc18)/np.mean(nEpc18):.2f} %', color=cm.coolwarm(0.))
    plt.stairs(H[0], H[1], color=cm.coolwarm(0.), linewidth=8)
    plt.xlabel(r'$\mathcal{N}_{\mathrm{Epc18}}/\hat{\mathcal{N}}_{\mathrm{Epc18}}$')
    plt.legend()
    plt.grid()
    
    plt.subplot(2, 4, 5)
    H = plt.hist(nIpc18/np.mean(nIpc18), bins=50, label=f'mean = {np.mean(nIpc18):.2f}, std = {np.std(nIpc18):.2f},\n relative std = {1e2*np.std(nIpc18)/np.mean(nIpc18):.2f} %', color=cm.coolwarm(0.))
    plt.stairs(H[0], H[1], color=cm.coolwarm(0.), linewidth=8)
    plt.xlabel(r'$\mathcal{N}_{\mathrm{Ipc18}}/\hat{\mathcal{N}}_{\mathrm{Ipc18}}$')
    plt.legend()
    plt.grid()
    
    plt.subplot(2, 4, 6)
    #plt.bar(['True', 'False'], [np.sum(valid), np.sum(valid == False)])
    #plt.xlabel('valid')
    #plt.yscale('log')
    #plt.grid()
    lr = fvalH0 - fval
    a, b, sigma = np.array(BB2DLLFiniteMC.computeSignificance(fvalH0, fval, 2, parametrizedX17=True, Ncrossing=0))
    if (lr < -1e-3).any():
        print('WARNING: lr < 0:', len(lr[lr < 0]), 'tests')
    sigma = sigma[lr > 0]
    
    # Compute median
    median = np.median(sigma)
    print(f'Files: {files}')
    print(f'Local median sigma: {median:.2f}\n')
    H = plt.hist(sigma, bins=50, label=f'mean = {np.mean(sigma):.2f}, std = {np.std(sigma):.2f}', color=cm.coolwarm(0.))
    plt.xlabel(r'Local significance [$\sigma$]')
    plt.stairs(H[0], H[1], color=cm.coolwarm(0.), linewidth=8)
    plt.legend()
    plt.grid()
    
    plt.subplot(2, 4, 7)
    lr = fvalH0 - fval
    a, b, sigma = np.array(BB2DLLFiniteMC.computeSignificance(fvalH0, fval, 2, parametrizedX17=True))
    if (lr < -1e-3).any():
        print('WARNING: lr < 0:', len(lr[lr < 0]), 'tests')
    sigma = sigma[lr > 0]
    H = plt.hist(sigma, bins=50, label=f'mean = {np.mean(sigma):.2f}, std = {np.std(sigma):.2f}', color=cm.coolwarm(0.))
    plt.stairs(H[0], H[1], color=cm.coolwarm(0.), linewidth=8)
    plt.xlabel(r'Global significance [$\sigma$]')
    plt.legend()
    plt.grid()
    
    # Compute median
    median = np.median(sigma)
    print(f'Files: {files}')
    print(f'Global median sigma: {median:.2f}\n')
    
    plt.subplot(2, 4, 8)
    if len(mX17) > 0:
        H = plt.hist(mX17, bins=50, label=f'mean = {np.mean(mX17):.2f}, std = {np.std(mX17):.2f}', color=cm.coolwarm(0.))
        plt.stairs(H[0], H[1], color=cm.coolwarm(0.), linewidth=8)
        plt.xlabel(r'$\mathrm{X17\,\,mass\,\,[MeV/c^2]}$')
        plt.legend()
        plt.grid()
    else:
        H = plt.hist(execTime, bins=50, label=f'mean = {np.mean(execTime):.2f}, std = {np.std(execTime):.2f}', color=cm.coolwarm(0.))
        plt.stairs(H[0], H[1], color=cm.coolwarm(0.), linewidth=8)
        plt.xlabel('Execution time [s]')
        plt.legend()
        plt.grid()

    
    plt.savefig(f'{prefix}.png', bbox_inches='tight')

    plt.figure(figsize=(42, 14), dpi=100)
    plt.suptitle(f'{prefix}, {len(nSig)} tests')
    plt.subplot(1, 3, 1)

    plt.hist(execTime, bins=50, label=f'mean = {np.mean(execTime):.2f}, std = {np.std(execTime):.2f}', color=cm.coolwarm(0.))
    plt.xlabel('Execution time')
    plt.legend()
    plt.grid()

    plt.subplot(1, 3, 2)
    plt.hist(execTimeH0, bins=50, label=f'mean = {np.mean(execTimeH0):.2f}, std = {np.std(execTimeH0):.2f}', color=cm.coolwarm(0.))
    plt.xlabel('Execution time H0')
    plt.legend()
    plt.grid()
    
    plt.subplot(1, 3, 3)
    plt.bar(['True', 'False'], [np.sum(valid), np.sum(valid == False)], color=cm.coolwarm(0.))
    plt.xlabel('valid')
    plt.yscale('log')
    plt.grid()

    plt.savefig(f'{prefix}_execTime.png', bbox_inches='tight')

