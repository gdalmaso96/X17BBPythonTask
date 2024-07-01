import numpy as np
from matplotlib.legend_handler import HandlerTuple
from matplotlib import pyplot as plt
import matplotlib
from matplotlib import cm
from scipy.stats import norm
import os
from scipy.interpolate import interp1d, RectBivariateSpline, RegularGridInterpolator
import time
startTime = time.time()

matplotlib.rcParams.update({'font.size': 30})
matplotlib.rcParams['figure.facecolor'] = 'white'

fileDir = "./results/"
fileDir = "./resultsMerlin/"
# Get list of files in the ./results/ folder
yields = [0, 100, 150, 200, 250, 300, 400, 450, 500, 600]
fractions = [0, 0.5, 1]
masses = np.arange(16.5, 17.1, 0.1)
_files = os.listdir(fileDir)
files = []
for y in yields:
    files = files + [f for f in _files if f.startswith(f"AlternativeFC_N{y}_p") and f.endswith(".txt")]


pX17 = []
mX17 = []
nX17 = []
lr = []

print('Elapsed time 1: %.2f s' %(time.time() - startTime))
for f in files:
    content = np.loadtxt(fileDir + f, dtype=str, skiprows=1)
    _n = content[:,0].astype(float)
    _p = content[:,1].astype(float)
    _m = content[:,2].astype(float)
    content = content[:, 3:]
    # TO BE FIXED
    _toy = content[:,2] == "True"
    _l = content[:,3].astype(float)
    _lc = content[:,4].astype(float)
    _lr = content[:,5].astype(float)
    _a = content[:,6] == "True"
    _v = content[:,7] == "True"
    
    # Take the toy data only
    pX17.append(_p[_toy])
    mX17.append(_m[_toy])
    nX17.append(_n[_toy])
    lr.append(_lr[_toy])
pX17 = np.concatenate(pX17)
mX17 = np.concatenate(mX17)
nX17 = np.concatenate(nX17)
lr = np.concatenate(lr)
ls = lr*(np.abs(lr) > 1e-5)

## Get all toy data sets
NSIGS = 0
PSIGS = 1
MSIGS = 17.0
files = [f for f in _files if f.startswith(f"DataAlternativeFC_N{NSIGS}_p") and f.endswith(".txt")]
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

fig = plt.figure(figsize=(28, 14))
plt.subplot(1, 2, 1)
plt.plot(DataNX17[SEED == 650], DataMX17[SEED == 650], 'o', label = 'Toy data sets')
plt.xlabel(r'$\mathcal{N}_{\mathrm{Sig}}$')
plt.ylabel(r'$m_{X17}$ [MeV/c$^2$]')
plt.grid()
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(DataNX17[SEED == 650], DataPX17[SEED == 650], 'o', label = 'Toy data sets')
plt.xlabel(r'$\mathcal{N}_{\mathrm{Sig}}$')
plt.ylabel(r'$p_{X17}$')
plt.grid()
plt.legend()


# Some jobs stopped before finishing and some seeds don't have a full grid
# Find them and remove them
# Create a mask for the seeds that have a full grid
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

fig = plt.figure(figsize=(14, 14))
plt.hist(Nsigs, histtype='step', color='r', label='Toy data sets')
fig = plt.figure(figsize=(14, 14))
plt.hist(Msigs, histtype='step', color='r', label='Toy data sets')
fig = plt.figure(figsize=(14, 14))
plt.hist(Psigs, histtype='step', color='r', label='Toy data sets')


# Get unique seeds
seed = np.unique(SEED)
masses = np.array([16.9])
fractions = np.array([0])

yields = np.unique(nX17)
yields = np.sort(yields)

CLs = []
UL = []
LL = []
print('Elapsed time 3: %.2f s' %(time.time() - startTime))
fig = plt.figure(figsize=(14,14), dpi = 100)
for s in seed:
    _cl = []
    for m in masses:
        for p in fractions:
            for y in yields:
                _temp = lr[(nX17 == y)*(np.abs(mX17 - m) < 1e-3)*(np.abs(pX17 - p) < 1e-3)]
                _cl.append(len(_temp[_temp < DataLr[(SEED == s)*(DataNX17 == y)*(np.abs(DataMX17 - m) < 1e-3)*(np.abs(DataPX17 - p) < 1e-3)]])/len(_temp))
    _cl = np.array(_cl)
    
    f = interp1d(yields, _cl, kind='linear')
    X = np.linspace(0, 600, 1000)
    Y = f(X)
    
    # Find UL
    tempY = 0
    tempX = 0
    ll = 0
    ul = 500
    for (yY, xX) in zip(Y, X):
        if tempY < 0.9 and yY >= 0.9:
            ul = xX
        elif tempY > 0.9 and yY <= 0.9:
            ll = xX
        tempX = xX
        tempY = yY
    UL.append(ul)
    LL.append(ll)
    
    if s == s0:
        plt.plot(yields, _cl, 'o-', color='b', zorder = 21)
        plt.plot(X, Y, '--', color='orange', zorder = 20)
        plt.vlines(ul, 0, 1, colors='g', linestyles='dashed', zorder = 22)
    plt.vlines(ll, 0, 1, colors='r', linestyles='dashed', alpha=0.1)
    plt.vlines(ul, 0, 1, colors='g', linestyles='dashed', alpha=0.1)
    CLs.append(_cl)

print('Datas:',len(CLs))
    
plt.hlines(0.9, 0, 600, colors='r', linestyles='dashed', label = '90% CL')
plt.xlim(0, 600)
plt.ylim(0, 1)
plt.xlabel(r'$\mathcal{N}_{\mathrm{Sig}}$')
plt.ylabel('CL')
plt.grid()
plt.legend()

fig = plt.figure(figsize=(14, 14), dpi=100)
plt.hist(UL, histtype='step', color='r', range = [0, 600], bins = 50, label=f'UL: {np.mean(UL):.1f} $\pm$ {np.std(UL):.1f}')
plt.hist(LL, histtype='step', color='b', range = [0, 600], bins = 50, label=f'LL: %.1f $\pm$ %.1f' %(np.mean(LL), np.std(LL)))

maxHeight = plt.gca().get_ylim()[1]
plt.vlines(np.median(UL), 0, maxHeight, colors='r', linestyles='dashed', label='Median UL = %.1f' %(np.median(UL)))
plt.vlines(np.median(LL), 0, maxHeight, colors='b', linestyles='dashed', label='Median LL = %.1f' %(np.median(LL)))
plt.grid()
plt.ylim(0, maxHeight)
plt.legend()
plt.xlabel(r'$\mathcal{N}_{\mathrm{Sig}}$ UL/LL (90% CL)')
plt.xlim(0, 600)



# Do 2D CLs
masses = np.unique(mX17)

CLs = []
Ms = []
Ns = []
UL = []
LL = []
massUL = []
massLL = []
fig = plt.figure(figsize=(14,14), dpi = 100)
ax = fig.add_subplot(111)
ax.plot(np.meshgrid(masses, yields)[0], np.meshgrid(masses, yields)[1], 'o', color='b', alpha=0.1, label='FC grid')
h = []
significance = []
print('Elapsed time 4: %.2f s' %(time.time() - startTime))
for s in seed:
    _cl = []
    tempS = 0
    for m in masses:
        for p in fractions:
            for y in yields:
                _temp = lr[(nX17 == y)*(np.abs(mX17 - m) < 1e-3)*(np.abs(pX17 - p) < 1e-3)]
                _cl.append(len(_temp[_temp < DataLr[(SEED == s)*(DataNX17 == y)*(np.abs(DataMX17 - m) < 1e-3)*(np.abs(DataPX17 - p) < 1e-3)]])/len(_temp))
                Ms.append(m)
                Ns.append(y)
                if y == 0:
                    tempS += _cl[-1]
    significance.append(tempS/len(masses)/len(fractions))
    
    f = RectBivariateSpline(masses, yields, np.array(_cl).reshape(len(masses), len(yields)), kx=1, ky=1)
    X = np.linspace(min(masses), max(masses), 500)
    Y = np.linspace(min(yields), max(yields), 1000)
    Z = f(X, Y).transpose()
    X,Y = np.meshgrid(X, Y)

    C = None
    if s%50 == 0:
        C = ax.contourf(X, Y, Z, levels=[0, 0.9], alpha=0.2, colors=[cm.coolwarm(s/max(seed))])
        _h, _l = C.legend_elements()
        h.append(_h[0])
        C = ax.contour(X, Y, Z, levels=[0.9], alpha=0.5, colors=[cm.coolwarm(s/max(seed))], linewidths=2)
    else:
        C = ax.contourf(X, Y, Z, levels=[0, 0.9], alpha=0, colors=[cm.coolwarm(s/max(seed))])
    plt.xlabel(r'$m_{X17}$ [MeV/c$^2$]')
    plt.ylabel(r'$\mathcal{N}_{\mathrm{Sig}}$')
    
    
    try:
        minX = X[Z < 0.9].min()
        maxX = X[Z < 0.9].max()
        minY = Y[Z < 0.9].min()
        maxY = Y[Z < 0.9].max()
        
        UL.append(maxY)
        LL.append(minY)
        massUL.append(maxX)
        massLL.append(minX)
    except:
        print('Error for SEED:', s)

significance = np.array(significance)
UL = np.array(UL)
LL = np.array(LL)
massUL = np.array(massUL)
massLL = np.array(massLL)
ax.legend()
_h, l = ax.get_legend_handles_labels()
print(l)
ax.legend([_h[0], tuple(h)], ['FC grid', 'Toy CI 90% CL'], handler_map={tuple: HandlerTuple(ndivide=None)})

fig = plt.figure(figsize=(14, 14), dpi=100)
plt.title(r'$\mathcal{N}_{\mathrm{Sig}} = $' + f'{NSIGS} UL/LL (90% CL)')
plt.hist(UL, histtype='step', color='r', bins = 50, label=f'UL: {np.mean(UL):.1f} $\pm$ {np.std(UL):.1f}')#, range = [0, 600])
plt.hist(LL, histtype='step', color='b', bins = 50, label=f'LL: %.1f $\pm$ %.1f' %(np.mean(LL), np.std(LL)))#, range = [0, 600])
plt.grid()
plt.xlabel(r'$\mathcal{N}_{\mathrm{Sig}}$ UL/LL (90% CL)')
#plt.xlim(0, 600)

plt.yscale('log')
maxHeight = plt.gca().get_ylim()[1]
minHeight = plt.gca().get_ylim()[0]
plt.vlines(np.median(UL), 0, maxHeight, colors='r', linestyles='dashed', label='Median UL = %.1f' %(np.median(UL)))
plt.vlines(np.median(LL), 0, maxHeight, colors='b', linestyles='dashed', label='Median LL = %.1f' %(np.median(LL)))
plt.legend()
plt.ylim(minHeight, maxHeight)

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
plt.xlim(16.5, 17.3)

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

print(len(UL), UL)
ll = LL[LL < NSIGS]
ul = UL[LL < NSIGS]
mll = massLL[LL < NSIGS]
mul = massUL[LL < NSIGS]
ll = ll[ul > NSIGS]
mll = mll[ul > NSIGS]
mul = mul[ul > NSIGS]
ll = ll[mll < 16.9]
mul = mul[mll < 16.9]
ll = ll[mul > 16.9]

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
plt.hlines(16.9, 0, 16.9, colors='r', linestyles='dashed')
plt.vlines(16.9, 16.9, 30, colors='r', linestyles='dashed')
plt.grid()
plt.xlabel('mass LL [90% CL]')
plt.ylabel('mass UL [90% CL]')
plt.xlim(16.5, 17.1)
plt.ylim(16.5, 17.1)

print('Elapsed time 5: %.2f s' %(time.time() - startTime))


# Do 3D CLs
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
    for m in masses:
        for p in fractions:
            for y in yields:
                _temp = lr[(nX17 == y)*(np.abs(mX17 - m) < 1e-3)*(np.abs(pX17 - p) < 1e-3)]
                _cl.append(len(_temp[_temp < DataLr[(SEED == s)*(DataNX17 == y)*(np.abs(DataMX17 - m) < 1e-3)*(np.abs(DataPX17 - p) < 1e-3)]])/len(_temp))
                Ps.append(p)
                Ms.append(m)
                Ns.append(y)
                if y == 0:
                    tempS += _cl[-1]
    significance.append(tempS/len(masses))
    
    points = (masses, fractions, yields)
    z = np.array(_cl).reshape(len(masses), len(fractions), len(yields))
    
    W = np.linspace(min(masses), max(masses), 100)
    X = np.linspace(min(fractions), max(fractions), 100)
    Y = np.linspace(min(yields), max(yields), 500)
    
    interp = RegularGridInterpolator((masses, fractions, yields), z, method='linear')
    
    W, X, Y = np.meshgrid(W, X, Y)
    Z = interp((W, X, Y))
    minW = W[Z < 0.9].min()
    maxW = W[Z < 0.9].max()
    minX = X[Z < 0.9].min()
    maxX = X[Z < 0.9].max()
    minY = Y[Z < 0.9].min()
    maxY = Y[Z < 0.9].max()
    
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

fig = plt.figure(figsize=(14, 14), dpi=100)
plt.title(r'$\mathcal{N}_{\mathrm{Sig}} = $' + f'{NSIGS} UL/LL (90% CL)')
plt.hist(UL, histtype='step', color='r', bins = 50, label=f'UL: {np.mean(UL):.1f} $\pm$ {np.std(UL):.1f}')#, range = [0, 600])
plt.hist(LL, histtype='step', color='b', bins = 50, label=f'LL: %.1f $\pm$ %.1f' %(np.mean(LL), np.std(LL)))#, range = [0, 600])
plt.grid()
plt.xlabel(r'$\mathcal{N}_{\mathrm{Sig}}$ UL/LL (90% CL)')
#plt.xlim(0, 600)

plt.yscale('log')
maxHeight = plt.gca().get_ylim()[1]
minHeight = plt.gca().get_ylim()[0]
plt.vlines(np.median(UL), 0, maxHeight, colors='r', linestyles='dashed', label='Median UL = %.1f' %(np.median(UL)))
plt.vlines(np.median(LL), 0, maxHeight, colors='b', linestyles='dashed', label='Median LL = %.1f' %(np.median(LL)))
plt.legend()
plt.ylim(minHeight, maxHeight)

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

fig = plt.figure(figsize=(14, 14), dpi=100)
plt.title(r'$p_{X17}$ UL/LL (90% CL)')
plt.hist(fractionUL, histtype='step', color='r', label=f'UL: {np.mean(fractionUL):.1f} $\pm$ {np.std(fractionUL):.1f}')
plt.hist(fractionLL, histtype='step', color='b', label=f'LL: %.1f $\pm$ %.1f' %(np.mean(fractionLL), np.std(fractionLL)))
plt.grid()
plt.xlabel(r'$p_{X17}$ UL/LL (90% CL)')
plt.xlim(0, 1)

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

#print(len(UL), UL)
ll = LL[LL < NSIGS]
ul = UL[LL < NSIGS]
mll = massLL[LL < NSIGS]
mul = massUL[LL < NSIGS]
ll = ll[ul > NSIGS]
mll = mll[ul > NSIGS]
mul = mul[ul > NSIGS]
ll = ll[mll < 16.9]
mul = mul[mll < 16.9]
ll = ll[mul > 16.9]

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
plt.hlines(16.9, 0, 16.9, colors='r', linestyles='dashed')
plt.vlines(16.9, 16.9, 30, colors='r', linestyles='dashed')
plt.grid()
plt.xlabel('mass LL [90% CL]')
plt.ylabel('mass UL [90% CL]')
plt.xlim(16.5, 17.1)
plt.ylim(16.5, 17.1)

print('Elapsed time 5: %.2f s' %(time.time() - startTime))
