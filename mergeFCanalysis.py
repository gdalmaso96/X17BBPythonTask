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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prefix', type=str, help='Prefix of the files to merge', default='fcAnalysis/bins20x14CurrentStatisticsParametrized_lratio_')
    parser.add_argument('-pl', '--plot', type=bool, help='Plot the number of toys per point', default=True)
    return parser.parse_args()

def mergeFiles(prefix, plot=False):
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
    print(N)
    M = np.unique(massX17Toy)
    print(M)
    
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
        plt.show()
        plt.savefig(prefix + 'nToys.png')
        
    print(data)
    # Compute the CLs values
    for i in range(len(N)):
        for j in range(len(M)):
            tempLR = lratio[(nX17Toy == N[i]) & (massX17Toy == M[j])]
            tempDATA = data[(nX17Toy == N[i]) & (massX17Toy == M[j])]
            # Arg sort and find the data index in the sorted array
            if len(tempLR) <= 1:
                continue
            idx = np.argsort(tempLR)
            idx = np.where(tempDATA[idx] == True)[0][0]
            #print(tempLR, tempDATA, idx)
            CLs[i][j] = float(idx) / float((len(tempLR) - 1))
    
    if plot:
        # Plot the CLs values
        plt.figure(figsize=(28, 14), dpi=100)
        plt.subplot(1, 2, 1)
        plt.title('Feldmann-Cousins')
        plt.imshow(CLs.transpose()[::-1], cmap=cm.coolwarm, extent=[N.min(), N.max(), M.min(), M.max()], aspect='auto')
        plt.colorbar()
        plt.ylim(16, 18)
        plt.ylabel('Mass [MeV]')
        plt.xlabel('nX17')
        
        # Draw 90% CL line
        plt.subplot(1, 2, 2)
        f = RectBivariateSpline(N, M, CLs)
        x = np.linspace(N.min(), N.max(), 100)
        y = np.linspace(M.min(), M.max(), 100)
        z = f(x, y).transpose()
        
        
        plt.imshow(z[::-1], cmap=cm.coolwarm, extent=[N.min(), N.max(), M.min(), M.max()], aspect='auto')
        plt.colorbar()
        plt.contour(x, y, z, colors='black', linewidths=5, levels=[0.9])
        plt.ylim(16, 18)
        
        plt.ylabel('Mass [MeV]')
        plt.xlabel('nX17')
        plt.title('Cubic interpolation')
        plt.savefig(prefix + 'CLs.png')
        plt.show()
    
    # Save the data
    np.savetxt(prefix + 'CLs.txt', CLs)
    np.savetxt(prefix + 'nToys.txt', nToys)

if __name__ == '__main__':
    args = parse_args()
    mergeFiles(args.prefix, args.plot)
