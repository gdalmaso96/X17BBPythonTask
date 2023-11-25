import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, RectBivariateSpline
from matplotlib import cm
import matplotlib
from scipy.stats import norm, chi2

matplotlib.rcParams.update({'font.size': 35})

# Load the data
fileName = 'results/BBstandard_profileLikelihood_SEED0.txt'
fileName = 'results/bins16x10CurrentStatisticsParametrized_profileLikelihood_SEED0.txt'

# Check number #
nHeaderLines = 0
with open(fileName, 'r') as f:
    for line in f:
        if line.startswith('#'):
            nHeaderLines += 1
        else:
            continue


if nHeaderLines == 1:
    # Check number of columns
    nColumns = len(np.loadtxt(fileName, unpack=True, skiprows=1))
    if nColumns == 2:
        nX17, nLL = np.loadtxt(fileName, unpack=True, skiprows=1)
        
        nX17 = np.array(nX17)
        nLL = np.array(nLL)
        nX17 = nX17[nLL < 1e7]
        nLL = nLL[nLL < 1e7]
        index = np.argsort(nX17)
        
        bestnX17 = nX17[0]
        bestnLL = nLL[0]
        nLL = nLL[index]
        nX17 = nX17[index]
        
        plt.figure(figsize=(14, 14), dpi=100)
        plt.title('Profile likelihood')
        plt.plot(nX17, nLL - np.min(nLL), 'o', linewidth=5, markersize=20, color = cm.coolwarm(0.))
        if len(nX17) > 4:
            x = np.linspace(nX17[0], nX17[-1], 1000)
            f = interp1d(nX17, nLL - np.min(nLL), kind='cubic')
            plt.plot(x, f(x), linewidth=5, color = cm.coolwarm(0.))
        plt.plot(bestnX17, bestnLL -np.min(nLL), 'o', linewidth=5, markersize=20, color = 'black', label='Best fit')
        plt.xlabel(f'number of signal events')
        plt.legend()
        #plt.ylim(0, None)
        plt.grid()
    else:
        nX17, mX17, LL = np.loadtxt(fileName, unpack=True, skiprows=1)
        
        bestLL = LL[0]
        bestnX17 = nX17[0]
        bestmX17 = mX17[0]
        nX17 = np.array(nX17)[1:]
        mX17 = np.array(mX17)[1:]
        LL = np.array(LL)[1:]
        print(bestnX17, bestmX17, bestLL)
        
        # Round to 2 decimals
        #nX17 = np.round(nX17, 2)
        #mX17 = np.round(mX17, 2)
        LL = np.array(LL)
        
        nX17 = nX17[LL < 1e7]
        mX17 = mX17[LL < 1e7]
        LL = LL[LL < 1e7]
        
        # Get unique values
        nX17u = np.unique(nX17)
        mX17u = np.unique(mX17)
        
        # Create 2D array
        LL2D = np.zeros((len(nX17u), len(mX17u)))
        
        # Fill 2D array
        for j in range(len(nX17u)):
            for i in range(len(mX17u)):
                print(nX17u[i], mX17u[j])
                print(nX17 == nX17u[i], mX17 == mX17u[j])
                index = np.logical_and(nX17 == nX17u[i], mX17 == mX17u[j])
                print(LL[index])
                LL2D[i][j] = LL[index]
        
        pvalue = 1 - chi2.sf(LL2D - bestLL, 2)
        
        # Interpolate
        f = RectBivariateSpline(nX17u, mX17u, pvalue)
        f1 = RectBivariateSpline(nX17u, mX17u, LL2D)
        nX17u = np.linspace(nX17u[0], nX17u[-1], 1000)
        mX17u = np.linspace(mX17u[0], mX17u[-1], 1000)
        pvalueT = f(nX17u, mX17u).transpose()
        LLT = f1(nX17u, mX17u).transpose()
        
        # Plot 2D array
        plt.figure(figsize=(28, 14), dpi=100)
        
        plt.subplot(121)
        plt.title('Profile likelihood ratio')
        plt.imshow(LL2D.transpose()[::-1] - bestLL, origin='lower', aspect='auto', extent=[nX17u[0], nX17u[-1], mX17u[0], mX17u[-1]], cmap='coolwarm')
        #plt.colorbar()
        plt.contour(nX17u, mX17u, LLT, levels=10, colors='black', linewidths=5, linestyles='dashed')
        
        plt.plot(bestnX17, bestmX17, '+', markeredgewidth=10, markersize=50, color = 'black', label='Best fit')
        plt.xlabel(f'number of signal events')
        plt.ylabel(f'X17 mass [MeV/c$^2$]')
        
        plt.subplot(122)
        plt.title('Local p-value')
        plt.imshow(pvalue.transpose()[::-1], origin='lower', aspect='auto', extent=[nX17u[0], nX17u[-1], mX17u[0], mX17u[-1]], cmap='coolwarm')
        plt.colorbar()
        plt.contour(nX17u, mX17u, pvalueT, levels=[0.68], colors='black', linewidths=5, linestyles='dashed', label='1$\sigma$')
        plt.contour(nX17u, mX17u, pvalueT, levels=[0.9], colors='black', linewidths=5, label='90 %')
        plt.plot(bestnX17, bestmX17, '+', markeredgewidth=10, markersize=50, color = 'black', label='Best fit')
        plt.legend()
        plt.xlabel(f'number of signal events')
        plt.ylabel(f'X17 mass [MeV/c$^2$]')
        
    
elif nHeaderLines == 2:
    header = 0
    nX17 = []
    mX17 = []
    nLL = []
    mLL = []
    with open(fileName, 'r') as f:
        for line in f:
            if line.startswith('#'):
                header += 1
            elif header == 1:
                line = line.split()
                nX17.append(float(line[0]))
                nLL.append(float(line[1]))
            elif header == 2:
                line = line.split()
                mX17.append(float(line[0]))
                mLL.append(float(line[1]))
    
    nX17 = np.array(nX17)
    nLL = np.array(nLL)
    nX17 = nX17[nLL < 0.5]
    nLL = nLL[nLL < 0.5]
    
    index = np.argsort(nX17)
    bestnX17 = nX17[0]
    bestnLL = nLL[0]
    nLL = np.array(nLL)[index]
    nX17 = np.array(nX17)[index]
    
    index = np.argsort(mX17)
    bestmX17 = mX17[0]
    bestmLL = mLL[0]
    mLL = np.array(mLL)[index]
    mX17 = np.array(mX17)[index]
    
    plt.figure(figsize=(28, 14), dpi=100)
    plt.suptitle('Profile likelihood')
    plt.subplot(121)
    plt.plot(nX17, nLL - np.min(nLL), 'o', linewidth=5, markersize=20, color = cm.coolwarm(0.))
    x = np.linspace(nX17[0], nX17[-1], 1000)
    f = interp1d(nX17, nLL - np.min(nLL), kind='cubic')
    plt.plot(x, f(x), linewidth=5, color = cm.coolwarm(0.))
    plt.plot(bestnX17, bestnLL -np.min(nLL), 'o', linewidth=5, markersize=20, color = 'black', label='Best fit')
    plt.xlabel(f'number of signal events')
    plt.ylim(0, None)
    plt.legend()
    plt.grid()
    
    plt.subplot(122)
    plt.plot(mX17, mLL - np.min(mLL), 'o', linewidth=5, markersize=20, color = cm.coolwarm(0.))
    x = np.linspace(mX17[0], mX17[-1], 1000)
    f = interp1d(mX17, mLL - np.min(mLL), kind='cubic')
    plt.plot(x, f(x), linewidth=5, color = cm.coolwarm(0.))
    plt.plot(bestmX17, bestmLL -np.min(mLL), 'o', linewidth=5, markersize=20, color = 'black', label='Best fit')
    plt.ylim(0, None)
    plt.xlabel(f'X17 mass [MeV/c$^2$]')
    plt.grid()