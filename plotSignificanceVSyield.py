import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib
from matplotlib import cm
from scipy.interpolate import CubicSpline
matplotlib.rcParams.update({'font.size': 35})
plt.rcParams['figure.constrained_layout.use'] = True

#filaName = 'VaryReferencebins20x14IdealStatisticsParametrized_Significance.txt'
filaName = 'VaryReferencebins20x14CurrentStatisticsParametrized_Significance.txt'

def plotSignificanceVSyield(filaName):
    # Search lines in file
    local = 'Local median sigma: '
    global_ = 'Global median sigma: '

    localS = []
    globalS = []
    X17 = []
    with open(filaName, 'r') as f:
        for line in f:
            if line.startswith(local):
                line = line.split()
                localS.append(float(line[3]))
            elif line.startswith(global_):
                line = line.split()
                globalS.append(float(line[3]))
            elif line.startswith('Files'):
                line = line.split()
                n = float(line[1][line[1].find('_')+1:line[1].find('_re')])
                if len(X17) == 0:
                    X17.append(n)
                elif n != X17[-1]:
                    X17.append(n)

    X17 = np.array(X17)
    localS = np.array(localS)
    globalS = np.array(globalS)            

    plt.figure(figsize=(14, 14), dpi=100)
    prefix = filaName
    plt.suptitle(f'Binning {prefix[prefix.find("bins") + 4:prefix.find("bins") + 9]}, {prefix[prefix.find("bins") + 9:prefix.find("Statistics")]} statistics')
    f = CubicSpline(X17, localS)
    plt.plot(X17, localS, 'o', linewidth=5, markersize=20, color = cm.coolwarm(0), label='Local significance')
    plt.plot(np.linspace(X17.min(), X17.max(), 1000), f(np.linspace(X17.min(), X17.max(), 1000)), linewidth=5, color = cm.coolwarm(0))
    f = CubicSpline(X17, globalS)
    plt.plot(X17, globalS, 'o', linewidth=5, markersize=20, color = cm.coolwarm(0.99), label='Global significance')
    plt.plot(np.linspace(X17.min(), X17.max(), 1000), f(np.linspace(X17.min(), X17.max(), 1000))*(f(np.linspace(X17.min(), X17.max(), 1000)) >0 ), linewidth=5, color = cm.coolwarm(0.99))
    plt.xlabel(r'$\hat{\mathcal{N}}_{\mathrm{Sig}}$')
    plt.ylabel(r'Significance [$\sigma$]')
    plt.ylim(0, 9.5)
    plt.grid()
    plt.legend()
    
    print(f(450))

    plt.savefig(prefix[:prefix.find('.txt')] + '.png', bbox_inches='tight')

if __name__ == '__main__':
    #argv = sys.argv[1:]
    argv = [filaName]
    
    if len(argv) == 1:
        plotSignificanceVSyield(argv[0])