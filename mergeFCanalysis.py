# This file merges all files from the FC analysis into one file with the local CLs values
import numpy as np
from matplotlib import pyplot as plt
from glob import glob
from matplotlib import cm
import matplotlib
from scipy.interpolate import RectBivariateSpline
import argparse
matplotlib.rcParams.update({'font.size': 35})
plt.rcParams['figure.constrained_layout.use'] = True

dataFile = 'results/bins20x14CurrentStatisticsParametrizedNullSig_profileLikelihood_SEED0.txt'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prefix', type=str, help='Prefix of the files to merge', default='fcAnalysis/bins20x14CurrentStatisticsParametrized_lratio_')
    parser.add_argument('-d', '--data', type=str, help='Path to the data file', default='')
    parser.add_argument('-ds', '--dataFiles', type=str, help='Path to list of data files', default='')
    parser.add_argument('-pl', '--plot', type=bool, help='Plot the number of toys per point', default=False)
    return parser.parse_args()

def mergeFiles(prefix, plot=False, dataFile=''):
    listOfFiles = glob(prefix + '*.txt')
    listOfFiles.sort()
    listOfFiles = [file for file in listOfFiles if not file.endswith('CLs.txt') and not file.endswith('nToys.txt')]
    
    # Create lists to store the data
    nX17Toy, massX17Toy, accurateH1, accurateH0, data, SEED, lratio = [], [], [], [], [], [], []
    
    # Loop over the files and store the data
    for file in listOfFiles:
        with open(file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                else:
                    line = line.split()
                    nX17Toy.append(float(line[0]))
                    massX17Toy.append(float(line[1]))
                    accurateH1.append(line[2] == 'True')
                    accurateH0.append(line[3] == 'True')
                    data.append(line[4] == 'True')
                    SEED.append(int(float(line[5])))
                    lratio.append(float(line[6]))
    
    # Convert the lists to numpy arrays
    nX17Toy = np.array(nX17Toy)
    massX17Toy = np.array(massX17Toy)
    accurateH1 = np.array(accurateH1)
    accurateH0 = np.array(accurateH0)
    data = np.array(data)
    SEED = np.array(SEED)
    lratio = np.array(lratio)
    
    # If external data is provided, add it to the lists and remove the previous data
    if dataFile != '':
        # Remove previous data
        nX17Toy = nX17Toy[data == False]
        massX17Toy = massX17Toy[data == False]
        accurateH1 = accurateH1[data == False]
        accurateH0 = accurateH0[data == False]
        SEED = SEED[data == False]
        lratio = lratio[data == False]
        data = data[data == False]
        # Check number #
        nHeaderLines = 0
        with open(dataFile, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    nHeaderLines += 1
                else:
                    continue
        if nHeaderLines > 1:
            print('Data file has more than one header line. Please check the file.')
            exit()
        
        tnSig, tmX17, tlr, tbest = np.loadtxt(dataFile, unpack=True, skiprows=2)
        #print(tnSig, tmX17, tlr-tbest)
        for (n, m, l) in zip(tnSig, tmX17, tlr):
            nX17Toy = np.append(nX17Toy, n)
            massX17Toy = np.append(massX17Toy, m)
            accurateH1 = np.append(accurateH1, True)
            accurateH0 = np.append(accurateH0, True)
            data = np.append(data, True)
            SEED = np.append(SEED, -1)
            lratio = np.append(lratio, l - tbest[0])
            #print(n, m, l - tbest[0])
    #print(data)
    # Check how many were not accurate
    print('Number of inaccurate H1: ', len(accurateH1[accurateH1 == False]))
    print('Number of inaccurate H0: ', len(accurateH0[accurateH0 == False]))
    
    # Remove the inaccurate ones
    #nX17Toy = nX17Toy[accurateH1]
    #massX17Toy = massX17Toy[accurateH1]
    #SEED = SEED[accurateH1]
    #lratio = lratio[accurateH1]
    #data = data[accurateH1]
    #accurateH0 = accurateH0[accurateH1]
    #accurateH1 = accurateH1[accurateH1]
    #
    #nX17Toy = nX17Toy[accurateH0]
    #massX17Toy = massX17Toy[accurateH0]
    #SEED = SEED[accurateH0]
    #lratio = lratio[accurateH0]
    #data = data[accurateH0]
    #accurateH1 = accurateH1[accurateH0]
    #accurateH0 = accurateH0[accurateH0]
    
    # Compute grid points
    N = np.unique(nX17Toy)
    #print(N)
    M = np.unique(massX17Toy)
    #print(M)
    
    # Create a 2D array to store the data
    CLs = np.ones((len(N), len(M)))
    
    # Number of Toys per point
    nToys = np.zeros((len(N), len(M)))
    for i in range(len(N)):
        for j in range(len(M)):
            nToys[i][j] = len(lratio[(nX17Toy == N[i]) & (massX17Toy == M[j])])
            
    if plot:
        # Plot the number of toys per point
        plt.figure()
        plt.imshow(nToys.transpose()[::-1], cmap=cm.coolwarm, extent=[N.min(), N.max(), M.min(), M.max()], aspect='auto')
        plt.colorbar()
        plt.ylabel('Mass [MeV]')
        plt.xlabel('nX17')
        plt.title('Number of toys per point')
        #plt.show()
        plt.savefig(prefix + 'nToys.png')
        
    #print(data)
    # Compute the CLs values
    for i in range(len(N)):
        for j in range(len(M)):
            tempLR = lratio[(abs(nX17Toy - N[i]) < 1e-3) & (abs(massX17Toy - M[j]) < 1e-4)]
            tempDATA = data[(abs(nX17Toy - N[i]) < 1e-3) & (abs(massX17Toy - M[j]) < 1e-4)]
            #print(tempLR, tempDATA, N[i], M[j])
            # Arg sort and find the data index in the sorted array
            if len(tempLR) <= 1:
                print('WARNING: less than 2 toys for point (%.1f, %.2f)' %(N[i], M[j]))
                continue
            idx = np.argsort(tempLR)
            idx = np.where(tempDATA[idx] == True)[0][0]
            #print(tempLR, tempDATA, idx)
            CLs[i][j] = float(idx) / float((len(tempLR) - 1))
    
    if plot:
        # Plot the CLs values
        plt.figure(figsize=(32, 14), dpi=100)
        ax = plt.subplot(1, 2, 1)
        plt.title('Feldmann-Cousins')
        plt.imshow(CLs.transpose()[::-1], cmap=cm.coolwarm, extent=[N.min(), N.max(), M.min(), M.max()], aspect='auto')
        plt.colorbar()
        # Minor ticks
        for i in range(len(N+1)):
            plt.plot([N.min() + i*(N.max() - N.min()) / (len(N)), N.min() + i*(N.max() - N.min()) / (len(N))], [M.min(), M.max()], color='white', linewidth=2)
        for i in range(len(M+1)):
            plt.plot([N.min(), N.max()], [M.min() + i*(M.max() - M.min()) / (len(M)), M.min() + i*(M.max() - M.min()) / (len(M))], color='white', linewidth=2)
        plt.grid(which='minor', color='w', linestyle='-', linewidth=2)
        plt.ylabel('Mass [MeV]')
        plt.xlabel('nX17')
        
        # Draw 90% CL line
        ax = plt.subplot(1, 2, 2)
        f = RectBivariateSpline(N, M, CLs)
        x = np.linspace(N.min(), N.max(), 100)
        y = np.linspace(M.min(), M.max(), 100)
        z = f(x, y).transpose()
        
        
        plt.imshow(z[::-1], cmap=cm.coolwarm, extent=[N.min(), N.max(), M.min(), M.max()], aspect='auto')
        plt.colorbar()
        cs = plt.contour(x, y, z, colors='black', linewidths=5, levels=[0.9])
        plt.ylim(15, 18)
        v = [cs.collections[0].get_paths()[i].vertices for i in range(len(cs.collections[0].get_paths()))]
        v = np.concatenate(v)
        x = v[:,0]
        y = v[:,1]
        
        nCLmin = x.min()
        nCLmax = x.max()
        mCLmin = y.min()
        mCLmax = y.max()
        print(nCLmin, nCLmax, mCLmin, mCLmax)
        if (f(0, y[np.argsort(y)]) < f(x.min(), y[np.argsort(y)])).any():
            nCLmin = 0
        
        h, _ = cs.legend_elements()
        textstr = r'CL 90%% on $\mathcal{N}_{\mathrm{Sig}}$: (%.1f, %.1f)' %(nCLmin, x.max())
        textstr = textstr + '\nCL 90%% on mass: (%.2f, %.2f) MeV/c' %(y.min(), y.max()) + r'$^{2}$'
        props = dict(boxstyle='round', facecolor='white', edgecolor='grey', alpha=0.7)
        ax.legend(h, [textstr], loc='lower right', fontsize=35)
        plt.ylabel('Mass [MeV]')
        plt.xlabel('nX17')
        plt.title('Cubic interpolation')
        if dataFile != '':
            plt.savefig(prefix + dataFile[dataFile.find('Null'):] + 'CLs.png')
        else:
            plt.savefig(prefix + 'CLs.png')
        #plt.show()
    
    # Save the data
    if dataFile != '':
        np.savetxt(prefix + dataFile[dataFile.find('Null'):] + 'CLs.txt', CLs)
        np.savetxt(prefix + dataFile[dataFile.find('Null'):] + 'nToys.txt', nToys)
    else:
        np.savetxt(prefix + 'CLs.txt', CLs)
        np.savetxt(prefix + 'nToys.txt', nToys)
    
    return nCLmin, nCLmax, mCLmin, mCLmax

if __name__ == '__main__':
    args = parse_args()
    
    print(args.dataFiles)
    if args.dataFiles != '':
        dataList = glob(args.dataFiles)
        dataList.sort()
        nCLmin = []
        nCLmax = []
        mCLmin = []
        mCLmax = []
        for data in dataList:
            try:
                print(data)
                nCLmin_, nCLmax_, mCLmin_, mCLmax_ = mergeFiles(args.prefix, args.plot, dataFile=data)
                nCLmin.append(nCLmin_)
                nCLmax.append(nCLmax_)
                mCLmin.append(mCLmin_)
                mCLmax.append(mCLmax_)
            except:
                continue
        plt.figure(figsize=(28, 28), dpi=100)
        prefix = args.prefix
        plt.suptitle(f'Binning {prefix[prefix.find("bins") + 4:prefix.find("bins") + 9]}, {prefix[prefix.find("bins") + 9:prefix.find("Statistics")]} statistics, ' + r'$\mathcal{N}_{\mathrm{Sig}}$' +  f' =  {dataList[0][dataList[0].find("Null") + 4: dataList[0].find("_prof")]}, {len(nCLmin)} toy MCs')
    
        plt.subplot(2, 2, 1)
        plt.hist(nCLmin, bins=25, label='median = %.2f' %np.median(nCLmin), color=cm.coolwarm(0.))
        plt.xlabel(r'Lower limit on $\mathcal{N}_{\mathrm{Sig}}$')
        plt.grid()
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.hist(nCLmax, bins=25, label='median = %.2f' %np.median(nCLmax), color=cm.coolwarm(0.))
        plt.xlabel(r'Upper limit on $\mathcal{N}_{\mathrm{Sig}}$')
        plt.grid()
        plt.legend()
        
        plt.subplot(2, 2, 3)
        plt.hist(mCLmin, bins=25, label='median = %.2f' %np.median(mCLmin), color=cm.coolwarm(0.))
        plt.xlabel('Lower limit on Mass [MeV/c$^2$]')
        plt.grid()
        plt.legend()
        
        plt.subplot(2, 2, 4)
        plt.hist(mCLmax, bins=25, label='median = %.2f' %np.median(mCLmax), color=cm.coolwarm(0.))
        plt.xlabel('Upper limit on Mass [MeV/c$^2$]')
        plt.grid()
        plt.legend()
        plt.savefig(args.prefix + dataList[0][dataList[0].find("Null"): dataList[0].find("_prof")] +'CLsHist.png', bbox_inches='tight')
    else:
        print(mergeFiles(args.prefix, args.plot, dataFile=args.data))
