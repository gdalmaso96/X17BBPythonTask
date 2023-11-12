import numpy as np
import matplotlib.pyplot as plt
from glob import glob

# Import all results files
results = glob('results/results_SEED*.txt')

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
accurate = []
fvalH0 = []
accurateH0 = []
lratio = []
pvalue = []
sigma = []
execTime = []
execTimeH0 = []

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
                accurate.append(bool(values[11]))
                fvalH0.append(float(values[12]))
                accurateH0.append(bool(values[13]))
                lratio.append(float(values[14]))
                pvalue.append(float(values[15]))
                sigma.append(float(values[16]))
                execTime.append(float(values[17]))
                execTimeH0.append(float(values[18]))
            
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
accurate = np.array(accurate)
fvalH0 = np.array(fvalH0)
accurateH0 = np.array(accurateH0)
lratio = np.array(lratio)
pvalue = np.array(pvalue)
sigma = np.array(sigma)
execTime = np.array(execTime)
execTimeH0 = np.array(execTimeH0)

# Plot results
plt.figure(figsize=(4*14, 2*14), dpi=100)
plt.subplot(2, 4, 1)
plt.hist(nSig, bins=100)
plt.xlabel('nSig')
plt.grid()

plt.subplot(2, 4, 2)
plt.hist(nEpc15, bins=100)
plt.xlabel('nEpc15')
plt.grid()

plt.subplot(2, 4, 3)
plt.hist(nIpc15, bins=100)
plt.xlabel('nIpc15')
plt.grid()

plt.subplot(2, 4, 4)
plt.hist(nEpc18, bins=100)
plt.xlabel('nEpc18')
plt.grid()

plt.subplot(2, 4, 5)
plt.hist(nIpc18, bins=100)
plt.xlabel('nIpc18')
plt.grid()

plt.subplot(2, 4, 6)
plt.bar(['True', 'False'], [np.sum(accurate), np.sum(accurate == False)])
plt.xlabel('accurate')
plt.grid()

plt.subplot(2, 4, 7)
plt.hist(sigma, bins=100)
plt.xlabel('sigma')
plt.grid()

plt.subplot(2, 4, 8)
plt.hist(execTime, bins=100)
plt.xlabel('Execution time')
plt.grid()

plt.savefig('results.png')