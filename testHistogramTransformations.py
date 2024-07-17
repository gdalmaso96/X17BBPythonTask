# This macro produces a histogram and applies a linear transformation to its bins.
import numpy as np
from scipy.interpolate import interp1d

# generate 2 histograms
histRight = np.histogram(np.random.normal(2, 1, 1000), bins=50, range=[-5, 5])
histLeft = np.histogram(np.random.normal(-2, 2, 1000), bins=50, range=[-5, 5])

# plot histogram
import matplotlib.pyplot as plt

plt.figure()
plt.hist(histRight[1][:-1], bins=histRight[1], weights=histRight[0], histtype='step')
plt.hist(histLeft[1][:-1], bins=histLeft[1], weights=histLeft[0], histtype='step')
plt.show()

c = 0.5

x = histRight[1][:-1] + (histRight[1][1] - histRight[1][0])/2

Hright = interp1d(x, histRight[0], kind='linear', fill_value=0, bounds_error=False)
Hleft = interp1d(x, histLeft[0], kind='linear', fill_value=0, bounds_error=False)

# plot interpolated histogram
plt.figure()
plt.hist(histRight[1][:-1], bins=histRight[1], weights=Hright(histRight[1][:-1]), histtype='step')
plt.hist(histLeft[1][:-1], bins=histLeft[1], weights=Hleft(histLeft[1][:-1]), histtype='step')
plt.show()

# Apply momentum morphing
muRight = np.average(histRight[1][:-1], weights=histRight[0])
sigmaRight = np.sqrt(np.average((histRight[1][:-1] - muRight)**2, weights=histRight[0]))
muLeft = np.average(histLeft[1][:-1], weights=histLeft[0])
sigmaLeft = np.sqrt(np.average((histLeft[1][:-1] - muLeft)**2, weights=histLeft[0]))

sigmaPrime = (1 - c)*sigmaRight + c*sigmaLeft
muPrime = (1 - c)*muRight + c*muLeft

aRight = sigmaRight/sigmaPrime
aLeft = sigmaLeft/sigmaPrime

bRight = muRight - aRight*muPrime
bLeft = muLeft - aLeft*muPrime

# Transform the two histograms along the bin axis
HrightNew = Hright(x + bRight)*aRight
HleftNew = Hleft(x + bLeft)*aLeft
print(np.sum(HrightNew))
print(np.sum(histRight[0]))

# plot interpolated histogram
plt.figure()
plt.hist(histRight[1][:-1], bins=histRight[1], weights=HrightNew, histtype='step')
plt.hist(histLeft[1][:-1], bins=histLeft[1], weights=HleftNew, histtype='step')
plt.show()

# new histogram
histFinal = (1 - c)*HrightNew + c*HleftNew

# plot interpolated histFinal with the orignals
plt.figure()
plt.hist(histRight[1][:-1], bins=histRight[1], weights=histRight[0], histtype='step')
plt.hist(histLeft[1][:-1], bins=histLeft[1], weights=histLeft[0], histtype='step')
plt.hist(histRight[1][:-1], bins=histRight[1], weights=histFinal, histtype='step')
plt.show()