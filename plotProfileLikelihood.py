import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib import cm
import matplotlib

matplotlib.rcParams.update({'font.size': 35})

# Load the data
fileName = 'results/BBstandard_profileLikelihood_SEED0.txt'

# Check number #
nHeaderLines = 0
with open(fileName, 'r') as f:
    for line in f:
        if line.startswith('#'):
            nHeaderLines += 1
        else:
            continue


if nHeaderLines == 1:
    nX17, nLL = np.loadtxt(fileName, unpack=True, skiprows=1)
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