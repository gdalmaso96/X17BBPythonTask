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

fileDir = "./resultsFinal/"
yields = np.concatenate((np.arange(0, 351, 25), np.arange(400, 651, 50)))
masses = np.arange(16.5, 17.1, 0.05)
fractions = np.arange(0, 1.01, 0.25)
_files = os.listdir(fileDir)
files = []
for y in yields:
    files = files + [f for f in _files if f.startswith(f"FC2023_N{y}_p") and f.endswith(".txt")]


pX17 = []
mX17 = []
nX17 = []
lr = []


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
    # TO BE FIXED
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
pX17 = np.concatenate(pX17)
mX17 = np.concatenate(mX17)
nX17 = np.concatenate(nX17)
lr   = np.concatenate(lr)
#ls = lr*(np.abs(lr) > 1e-5)

## Get all toy data sets
DataPX17 = np.array(DataPX17)
DataMX17 = np.array(DataMX17)
DataNX17 = np.array(DataNX17)
DataLr = np.array(DataLr)
#Nsigs = np.concatenate(Nsigs)
#Msigs = np.concatenate(Msigs)
#Psigs = np.concatenate(Psigs)


# Compute limits
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
            
            #fig = plt.figure(figsize=(21,14), dpi = 100)
            #ax = fig.add_subplot(111)
            #DATALR = _lr
            #plt.title('Mass = %.2f, Fraction = %.2f, Yield = %d' %(m, p, y))
            #ax.hist(_temp, bins=100, color='b', alpha=0.5, label='FC grid')
            #ax.vlines(DATALR, 0, 1000, color='r', label='CL = %.2f %%' %(_cl[-1]*100))
            #plt.yscale('log')
            #plt.grid()
            #plt.legend()
            #plt.show()
            
significance.append(tempS/counterS)

points = (masses, fractions, yields)
z = np.array(_cl).reshape(len(masses), len(fractions), len(yields))

W = np.linspace(min(masses), max(masses), 100)
X = np.linspace(min(fractions), max(fractions), 100)
Y = np.linspace(min(yields), max(yields), 500)

interp = RegularGridInterpolator((masses, fractions, yields), z, method='linear')

W, X, Y = np.meshgrid(W, X, Y)
Z = interp((W, X, Y))
minW = W[Z <= 0.9].min()
maxW = W[Z <= 0.9].max()
minX = X[Z <= 0.9].min()
maxX = X[Z <= 0.9].max()
minY = Y[Z <= 0.9].min()
maxY = Y[Z <= 0.9].max()
DataUL = maxY

significance = np.array(significance)
plt.show()


# Plot lr distribution for y = 0
fig = plt.figure(figsize=(21,14), dpi = 100)
ax = fig.add_subplot(111)

DATALR = DataLr[(DataNX17 == 0)][0]

print('How many zeros within tolerance?', 1*(lr[(nX17 == 0)] < 1e-10).sum()/len(lr[(nX17 == 0)])*100)
ax.hist(lr[(nX17 == 0)], bins=100, color='b', alpha=0.5, label='FC grid')
ax.vlines(DATALR, 0, 1000, color='r', label='CL = %.2f %%' %(len(lr[(nX17 == 0)*(lr < DATALR)])/len(lr[(nX17 == 0)])*100))
plt.yscale('log')
plt.grid()
plt.legend()
plt.show()

print('Elapsed time 5: %.2f s' %(time.time() - startTime))

fig = plt.figure(figsize=(21,14), dpi = 100)
ax = fig.add_subplot(111)
plt.hist(numberOfToys, bins=100, color='b', alpha=0.5)
plt.grid()
plt.show()

# Plot Limits
# Plot voxel contours and projections

fig = plt.figure(figsize=(42,14), dpi = 100)
plt.subplots_adjust(wspace=0.5)
ax = fig.add_subplot(131, projection='3d')

plt.title('3D 90 % CL, ' + r'$\mathcal{N}_{\mathrm{Sig}}$' + f' UL = {maxY:.0f}')

limits = Z <= 0.9

# Get the corners of the voxel
Wedge = np.linspace(min(masses) - 0.5*(max(masses) - min(masses))/100, max(masses) + 0.5*(max(masses) - min(masses))/100, 101)
Xedge = np.linspace(min(fractions) - 0.5*(max(fractions) - min(fractions))/100, max(fractions) + 0.5*(max(fractions) - min(fractions))/100, 101)
Yedge = np.linspace(min(yields) - 0.5*(max(yields) - min(yields))/500, max(yields) + 0.5*(max(yields) - min(yields))/500, 501)

Wedge, Xedge, Yedge = np.meshgrid(Wedge, Xedge, Yedge)

colors = cm.coolwarm(Z)

edgecolors = np.clip(2*colors - 0.5, 0, 1)

# Color based on Z
ax.voxels(Yedge, Xedge, Wedge, limits, facecolors=colors, edgecolors=edgecolors)
# Adjust label distance from axis
ax.set_xlabel('Yield', labelpad=20)
ax.set_ylabel('Fraction', labelpad=20)
ax.set_zlabel('Mass', labelpad=40)
ax.set_aspect('auto')
# Invert Y axis
#ax.invert_yaxis()
plt.xlim(minY, maxY)
plt.ylim(minX, maxX)

# Rotate around the z axis by 90 deg
ax.view_init(elev=30, azim=45)

plt.show()


print('Elapsed time 7: %.2f s' %(time.time() - startTime))

### Get all toy data sets
NSIGS = 0
MSIGS = 16.97
files = [f for f in _files if f.startswith(f"DataFC2023_N{NSIGS}_p") and f.endswith(".txt")]
DataPX17 = []
DataMX17 = []
DataNX17 = []
DataLr = []
Nsigs = []
Psigs = []
Msigs = []
SEED = []

print('Elapsed time 2: %.2f s' %(time.time() - startTime))
for f in files:
    content = np.loadtxt(fileDir + f, dtype=str, skiprows=1)
    content = content[:, 3:]
    _n = content[:,0].astype(float)
    _p = content[:,1].astype(float)
    _m = content[:,2].astype(float)
    Nsigs.append(content[:,3].astype(float))
    Psigs.append(content[:,4].astype(float))
    Msigs.append(content[:,5].astype(float))
    content = content[:, 3:]
    _l = content[:,3].astype(float)
    _lc = content[:,4].astype(float)
    _lr = content[:,5].astype(float)
    _a = content[:,6].astype('|S5')
    _v = content[:,7].astype('|S5')
    SEED.append(content[:,8].astype(int))
    
    DataPX17.append(_p)
    DataMX17.append(_m)
    DataNX17.append(_n)
    DataLr.append(_lr)

DataPX17 = np.concatenate(DataPX17)
DataMX17 = np.concatenate(DataMX17)
DataNX17 = np.concatenate(DataNX17)
DataLr = np.concatenate(DataLr)
Nsigs = np.concatenate(Nsigs)
Msigs = np.concatenate(Msigs)
Psigs = np.concatenate(Psigs)
SEED = np.concatenate(SEED)
seed = np.unique(SEED)

fig = plt.figure(figsize=(28, 14))
plt.subplot(1, 2, 1)
plt.plot(DataNX17[SEED == 0], DataMX17[SEED == 0], 'o', label = 'Toy data sets')
plt.xlabel(r'$\mathcal{N}_{\mathrm{Sig}}$')
plt.ylabel(r'$m_{X17}$ [MeV/c$^2$]')
plt.grid()
#plt.legend()

plt.subplot(1, 2, 2)
plt.plot(DataNX17[SEED == 0], DataPX17[SEED == 0], 'o', label = 'Toy data sets')
plt.xlabel(r'$\mathcal{N}_{\mathrm{Sig}}$')
plt.ylabel(r'$p_{X17}$')
plt.grid()
#plt.legend()

plt.show()

CorrectlyGenerated = []
length = len(np.unique(DataNX17))*len(np.unique(DataMX17))*len(np.unique(DataPX17))
print(length)
for s in np.unique(SEED):
    #print(len(DataLr[SEED == s]), s)
    if len(DataLr[SEED == s]) == length: # and s < 100:
        CorrectlyGenerated.append(s)

DataPX17 = DataPX17[np.isin(SEED, CorrectlyGenerated)]
DataMX17 = DataMX17[np.isin(SEED, CorrectlyGenerated)]
DataNX17 = DataNX17[np.isin(SEED, CorrectlyGenerated)]
DataLr = DataLr[np.isin(SEED, CorrectlyGenerated)]
Nsigs = Nsigs[np.isin(SEED, CorrectlyGenerated)]
Msigs = Msigs[np.isin(SEED, CorrectlyGenerated)]
Psigs = Psigs[np.isin(SEED, CorrectlyGenerated)]
SEED = SEED[np.isin(SEED, CorrectlyGenerated)]
s0 = min(np.unique(SEED))

Nsigs = Nsigs[DataNX17 == 0]
fractions = np.unique(pX17)

CLs = []
Ps = []
Ms = []
Ns = []
UL = []
LL = []
fractionUL = []
fractionLL = []
massUL = []
massLL = []
fig = plt.figure(figsize=(28,14), dpi = 100)
ax = fig.add_subplot(121)
ax.plot(np.meshgrid(masses, yields)[0], np.meshgrid(masses, yields)[1], 'o', color='b', alpha=0.1, label='FC grid')
ax = fig.add_subplot(122)
ax.plot(np.meshgrid(masses, fractions)[0], np.meshgrid(masses, fractions)[1], 'o', color='b', alpha=0.1, label='FC grid')
h = []
significance = []
print('Elapsed time 6: %.2f s' %(time.time() - startTime))
for s in seed:
    print('SEED:', s, '-', '%.2f' %(time.time() - startTime))
    _cl = []
    tempS = 0
    counterS = 0
    for m in masses:
        for p in fractions:
            for y in yields:
                _temp = lr[(nX17 == y)*(np.abs(mX17 - m) < 1e-2)*(np.abs(pX17 - p) < 1e-2)]
                _cl.append(len(_temp[_temp <= DataLr[(SEED == s)*(DataNX17 == y)*(np.abs(DataMX17 - m) < 1e-2)*(np.abs(DataPX17 - p) < 1e-2)]])/len(_temp))
                Ps.append(p)
                Ms.append(m)
                Ns.append(y)
                if y == 0:
                    tempS += _cl[-1]
                    counterS += 1
    significance.append(tempS/counterS)
    
    points = (masses, fractions, yields)
    z = np.array(_cl).reshape(len(masses), len(fractions), len(yields))
    
    W = np.linspace(min(masses), max(masses), 100)
    X = np.linspace(min(fractions), max(fractions), 100)
    Y = np.linspace(min(yields), max(yields), 500)
    
    interp = RegularGridInterpolator((masses, fractions, yields), z, method='linear')
    
    W, X, Y = np.meshgrid(W, X, Y)
    Z = interp((W, X, Y))
    minW = W[Z <= 0.9].min()
    maxW = W[Z <= 0.9].max()
    minX = X[Z <= 0.9].min()
    maxX = X[Z <= 0.9].max()
    minY = Y[Z <= 0.9].min()
    maxY = Y[Z <= 0.9].max()
    
    UL.append(maxY)
    LL.append(minY)
    massUL.append(maxW)
    massLL.append(minW)
    fractionUL.append(maxX)
    fractionLL.append(minX)

significance = np.array(significance)
UL = np.array(UL)
LL = np.array(LL)
massUL = np.array(massUL)
massLL = np.array(massLL)
fractionUL = np.array(fractionUL)
fractionLL = np.array(fractionLL)
plt.show()

fig = plt.figure(figsize=(14, 14), dpi=100)
plt.title(r'$\mathcal{N}_{\mathrm{Sig}} = $' + f'{NSIGS} UL/LL (90% CL)')
plt.hist(UL, histtype='step', color='r', bins = 50, label=f'UL: {np.mean(UL):.1f} $\pm$ {np.std(UL):.1f}', linewidth=3)
plt.hist(LL, histtype='step', color='b', bins = 50, label=f'LL: %.1f $\pm$ %.1f' %(np.mean(LL), np.std(LL)), linewidth=3)
plt.grid()
plt.xlabel(r'$\mathcal{N}_{\mathrm{Sig}}$ UL/LL (90% CL)')
#plt.xlim(0, 600)

plt.yscale('log')
maxHeight = plt.gca().get_ylim()[1]
minHeight = plt.gca().get_ylim()[0]
plt.vlines(np.median(UL), 0, maxHeight, colors='r', linestyles='dashed', label='Median UL = %.1f' %(np.median(UL)))
plt.vlines(np.median(LL), 0, maxHeight, colors='b', linestyles='dashed', label='Median LL = %.1f' %(np.median(LL)))
plt.vlines(DataUL, 0, maxHeight, colors='k', linestyles='dashed', label=f'Data: {len(UL[UL > DataUL])/len(UL)*100: .1f}% toys above data')
plt.legend()
plt.ylim(minHeight, maxHeight)
plt.show()

fig = plt.figure(figsize=(14, 14), dpi=100)
plt.title(r'$m_{X17}$ UL/LL (90% CL)')
plt.hist(massUL, histtype='step', color='r', label=f'UL: {np.mean(massUL):.1f} $\pm$ {np.std(massUL):.1f}')
plt.hist(massLL, histtype='step', color='b', label=f'LL: %.1f $\pm$ %.1f' %(np.mean(massLL), np.std(massLL)))
plt.grid()
plt.xlabel(r'$m_{X17}$ UL/LL (90% CL)')
plt.xlim(0, 30)

maxHeight = plt.gca().get_ylim()[1]
plt.vlines(np.median(massUL), 0, maxHeight, colors='r', linestyles='dashed', label='Median UL = %.1f' %(np.median(massUL)))
plt.vlines(np.median(massLL), 0, maxHeight, colors='b', linestyles='dashed', label='Median LL = %.1f' %(np.median(massLL)))
plt.legend()
plt.ylim(0, maxHeight)
plt.xlim(16.5, 17.1)
plt.show()

fig = plt.figure(figsize=(14, 14), dpi=100)
plt.title(r'$p_{X17}$ UL/LL (90% CL)')
plt.hist(fractionUL, histtype='step', color='r', label=f'UL: {np.mean(fractionUL):.1f} $\pm$ {np.std(fractionUL):.1f}')
plt.hist(fractionLL, histtype='step', color='b', label=f'LL: %.1f $\pm$ %.1f' %(np.mean(fractionLL), np.std(fractionLL)))
plt.grid()
plt.xlabel(r'$p_{X17}$ UL/LL (90% CL)')
plt.xlim(0, 1)
plt.show()

print(f'Elapsed time: {time.time() - startTime:.2f} s')

significance = 1 - significance
fig = plt.figure(figsize=(14, 14), dpi=100)
plt.hist(significance, histtype='step', color='r', label=f'p-value: {np.mean(significance):.2e} $\pm$ {np.std(significance):.1f}', bins=50, range=[0, 1])
maxHeight = plt.gca().get_ylim()[1]
plt.vlines(np.median(significance), 0, maxHeight, colors='r', linestyles='dashed', label='Median p-value = %.2e' %(np.median(significance)))
plt.grid()
plt.xlabel(r'p-value')
plt.ylim(0, maxHeight)
plt.xlim(0, 1)
plt.legend()
plt.show()

fig = plt.figure(figsize=(14, 14), dpi=100)
zs = norm.isf(significance)
zs = zs[np.isfinite(zs)]
plt.hist(zs, histtype='step', color='r', label=f'Significance: {np.mean(zs):.1f} $\pm$ {np.std(zs):.1f}', bins=50, range=[-4 + np.mean(zs), 4 + np.mean(zs)])
maxHeight = plt.gca().get_ylim()[1]
plt.vlines(np.median(zs), 0, maxHeight, colors='r', linestyles='dashed', label='Median significance = %.1f' %(np.median(zs)) + r' $\sigma$')
plt.grid()
plt.xlabel(r'Z-score')
plt.ylim(0, maxHeight)
plt.legend()
plt.show()

#print(len(UL), UL)
ll = LL[LL < NSIGS]
ul = UL[LL < NSIGS]
mll = massLL[LL < NSIGS]
mul = massUL[LL < NSIGS]
ll = ll[ul > NSIGS]
mll = mll[ul > NSIGS]
mul = mul[ul > NSIGS]
ll = ll[mll < MSIGS]
mul = mul[mll < MSIGS]
ll = ll[mul > MSIGS]

print('ULs above data:', len(UL[UL > DataUL])/len(UL)*100)

Coverage = len(ll)/len(LL)*100

fig = plt.figure(figsize=(28, 14), dpi=100)
plt.subplot(1, 2, 1)
plt.plot(LL, UL, '.', label=f'{Coverage:.1f}% coverage')
plt.xlabel(r'$\mathcal{N}$' + ' LL [90% CL]')
plt.ylabel(r'$\mathcal{N}$' + 'UL [90% CL]')

XMIN = np.min(LL)
XMIN = XMIN - 50 - XMIN%50
XMAX = np.max(LL)
XMAX = XMAX + 50 + (50 - XMAX%50)
YMIN = np.min(UL)
YMIN = YMIN - 50 - YMIN%50
YMAX = np.max(UL)
YMAX = YMAX + 50 + (50 - YMAX%50)
XMIN = max(0, XMIN)
plt.hlines(NSIGS, XMIN, NSIGS, colors='r', linestyles='dashed')
plt.vlines(NSIGS, NSIGS, YMAX, colors='r', linestyles='dashed')
plt.grid()
plt.legend()
plt.xlim(XMIN, XMAX)
plt.ylim(YMIN, YMAX)

plt.subplot(1, 2, 2)
plt.plot(massLL, massUL, '.')
plt.hlines(MSIGS, 0, MSIGS, colors='r', linestyles='dashed')
plt.vlines(MSIGS, MSIGS, 30, colors='r', linestyles='dashed')
plt.grid()
plt.xlabel('mass LL [90% CL]')
plt.ylabel('mass UL [90% CL]')
plt.xlim(16.5, 17.1)
plt.ylim(16.5, 17.1)
plt.show()

print('Probability of getting a lower limit:', np.sum(LL > 0)/len(LL))
print('Probability of observing a 3 sigma signal:', np.sum(zs > 3)/len(zs))

print('Elapsed time 5: %.2f s' %(time.time() - startTime))
