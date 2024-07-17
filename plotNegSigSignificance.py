import numpy as np
from matplotlib.legend_handler import HandlerTuple
from matplotlib import pyplot as plt
import matplotlib
from matplotlib import cm
from scipy.stats import norm
import os
from scipy.interpolate import interp1d, RectBivariateSpline, RegularGridInterpolator
import time
from matplotlib.colors import Normalize


def imshow3d(ax, array, value_direction='z', pos=0, norm=None, cmap=None):
    """
    Display a 2D array as a  color-coded 2D image embedded in 3d.

    The image will be in a plane perpendicular to the coordinate axis *value_direction*.

    Parameters
    ----------
    ax : Axes3D
        The 3D Axes to plot into.
    array : 2D numpy array
        The image values.
    value_direction : {'x', 'y', 'z'}
        The axis normal to the image plane.
    pos : float
        The numeric value on the *value_direction* axis at which the image plane is
        located.
    norm : `~matplotlib.colors.Normalize`, default: Normalize
        The normalization method used to scale scalar data. See `imshow()`.
    cmap : str or `~matplotlib.colors.Colormap`, default: :rc:`image.cmap`
        The Colormap instance or registered colormap name used to map scalar data
        to colors.
    """
    if norm is None:
        norm = Normalize()
    colors = plt.get_cmap(cmap)(norm(array))

    if value_direction == 'x':
        nz, ny = array.shape
        zi, yi = np.mgrid[0:nz + 1, 0:ny + 1]
        xi = np.full_like(yi, pos)
    elif value_direction == 'y':
        nx, nz = array.shape
        xi, zi = np.mgrid[0:nx + 1, 0:nz + 1]
        yi = np.full_like(zi, pos)
    elif value_direction == 'z':
        ny, nx = array.shape
        yi, xi = np.mgrid[0:ny + 1, 0:nx + 1]
        zi = np.full_like(xi, pos)
    else:
        raise ValueError(f"Invalid value_direction: {value_direction!r}")
    ax.plot_surface(xi, yi, zi, rstride=1, cstride=1, facecolors=colors, shade=False)
    
startTime = time.time()

matplotlib.rcParams.update({'font.size': 30})
matplotlib.rcParams['figure.facecolor'] = 'white'

fileDir = "./resultsFinalNegSig/"
#fileDir = "./resultsFinalNegSigAntoine/"
yields = np.array([0])
masses = np.arange(16.5, 17.3, 0.05)
fractions = np.arange(0, 1.01, 0.25)
_files = os.listdir(fileDir)
files = []
for y in yields:
    files = files + [f for f in _files if f.startswith(f"FC2023_negSignal_N{y}_p") and f.endswith(".txt")]


pX17 = []
mX17 = []
nX17 = []
lr = []

Nfit = []
Pfit = []
Mfit = []
DataPX17 = []
DataMX17 = []
DataNX17 = []
DataLr = []
Nsigs = []
Psigs = []
Msigs = []
print('Elapsed time 1: %.2f s' %(time.time() - startTime))
for f in files:
    content = np.loadtxt(fileDir + f, dtype=str, skiprows=1)
    if len(content.shape) == 1:
        continue
    DataNX17.append((content[:,0].astype(float))[0])
    DataPX17.append((content[:,1].astype(float))[0])
    DataMX17.append((content[:,2].astype(float))[0])
    DataLr.append((content[:,9].astype(float))[0])
    content = content[1:]
    _n = content[:,0].astype(float)
    _p = content[:,1].astype(float)
    _m = content[:,2].astype(float)
    content = content[:, 3:]
    nFit = content[:,0].astype(float)
    pFit = content[:,1].astype(float)
    mFit = content[:,2].astype(float)
    _toy = content[:,3] == "True"
    _l = content[:,4].astype(float)
    _lc = content[:,5].astype(float)
    _lr = content[:,6].astype(float)
    _a = content[:,7] == "True"
    _v = content[:,8] == "True"
    
    _p = _p[_toy]
    _m = _m[_toy]
    _n = _n[_toy]
    _l = _l[_toy]
    _lc = _lc[_toy]
    _lr = _lr[_toy]
    _a = _a[_toy]
    _v = _v[_toy]
    
    # Take the toy data only
    pX17.append(_p[_lr > -1e-5])
    mX17.append(_m[_lr > -1e-5])
    nX17.append(_n[_lr > -1e-5])
    lr.append(_lr[_lr > -1e-5])
    Nfit.append(nFit)
    Pfit.append(pFit)
    Mfit.append(mFit)
pX17 = np.concatenate(pX17)
mX17 = np.concatenate(mX17)
nX17 = np.concatenate(nX17)
lr   = np.concatenate(lr)
Nfit = np.concatenate(Nfit)
Pfit = np.concatenate(Pfit)
Mfit = np.concatenate(Mfit)
#ls = lr*(np.abs(lr) > 1e-5)

## Get all toy data sets
DataPX17 = np.array(DataPX17)
DataMX17 = np.array(DataMX17)
DataNX17 = np.array(DataNX17)
DataLr = np.array(DataLr)
#Nsigs = np.concatenate(Nsigs)
#Msigs = np.concatenate(Msigs)
#Psigs = np.concatenate(Psigs)


Ps = []
Ms = []
Ns = []
fig = plt.figure(figsize=(28,14), dpi = 100)
ax = fig.add_subplot(121)
ax.plot(np.meshgrid(masses, yields)[0], np.meshgrid(masses, yields)[1], 'o', color='b', alpha=0.1, label='FC grid')
ax = fig.add_subplot(122)
ax.plot(np.meshgrid(masses, fractions)[0], np.meshgrid(masses, fractions)[1], 'o', color='b', alpha=0.1, label='FC grid')
h = []
significance = []

print('Elapsed time 6: %.2f s' %(time.time() - startTime))

_cl = []
tempS = 0
counterS = 0
numberOfToys = []
for m in masses:
    for p in fractions:
        for y in yields:
            _temp = lr[(nX17 == y)*(np.abs(mX17 - m) < 1e-2)*(np.abs(pX17 - p) < 1e-2)]
            _lr = DataLr[(DataNX17 == y)*(np.abs(DataMX17 - m) < 1e-2)*(np.abs(DataPX17 - p) < 1e-2)]
            try:
                _lr = _lr[0]
            except:
                _lr = _lr
            _cl.append(len(_temp[_temp <= _lr])/len(_temp))
            Ps.append(p)
            Ms.append(m)
            Ns.append(y)
            if y == 0:
                _cl[-1] = 0
                tempS += _cl[-1]
                counterS += 1
            
            if y != 0:
                numberOfToys.append(len(_temp))

significance.append(tempS/counterS)
            
fig = plt.figure(figsize=(14,14), dpi = 100)
#plt.title('Null hypothesis likelihood ratio, negative signal allowed')
ax = fig.add_subplot(111)
DATALR = 27.248333632267304
ax.hist(lr, bins=100, color='b', alpha=0.5, label='Toys above data: %.2f %%' %(len(lr[lr >= DATALR])/len(lr)*100))
ax.vlines(DATALR, 0, 100000, color='r', label='Data')

plt.ylim(0.5, 2e4)
print('Number toys: %d' %(len(lr)))
print('Number toys >= DATALR: %d' %(len(lr[lr >= DATALR])))
plt.xlabel('Likelihood ratio')
plt.yscale('log')
plt.grid()
plt.legend()
plt.savefig('significance.png', bbox_inches='tight')
plt.show() 

print('Elapsed time 7: %.2f s' %(time.time() - startTime))

print('Sigmas: %.2f' %(norm.isf(len(lr[lr >= DATALR])/len(lr))))

fig = plt.figure(figsize=(42,14), dpi = 100)
plt.suptitle('Toy estimators distributions - generation from negative signal fit')
plt.subplot(131)
plt.hist(Nfit, bins=100, color='b', alpha=0.5, label='Toys')
plt.xlabel(r'$\mathcal{N}_{\mathrm{Sig}}$')


plt.subplot(132)
plt.hist(Pfit, bins=100, color='b', alpha=0.5, label='Toys')
plt.xlabel(r'$\mathcal{p}_{\mathrm{Sig18.1}}$')


plt.subplot(133)
plt.hist(Mfit, bins=100, color='b', alpha=0.5, label='Toys')
plt.xlabel(r'$\mathcal{m}_{\mathrm{Sig18.1}}$')
