import uproot
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

np.random.seed(10)

# Invariant mass
me = 0.00051099895000 # GeV

mIPC15min = 7.5
mIPC15max = 25

mIPC18min = 11
mIPC18max = 25

mX17min = 11
mX17max = 25

tIPC15min = 20
tIPC15max = 180

tIPC18min = 20
tIPC18max = 180

tX17min = 20
tX17max = 180

imasIpc15 = [-1.20088328e-01, -6.96710143e-03, 1.36576797e+01, 1.00809056e-01, 2.95068195e-03, 2.20480653e-02, 1.84422865e+00, 6.19679828e-06, 1.61736263e-09, 2.83566606e-01]
imasIpc18 = [1.96581337e+00, -3.21265500e-01, -5.48888410e-08, 1.67391473e+01, 2.09329009e-01, 1.99442656e+01, 1.99666205e+01, 1.08793476e+00, 2.24068933e-07, 6.66389915e-09, 8.17353976e-01]
imasX17   = [1.50422003e+01, 1.03486878e+00, 2.98311392e-07, 3.21431770e-10, 1.00014831e-01, 1.46389438e-07, 1.32648149e-02, 1.65414058e+01, 6.29824013e-01, 2.77675866e-04, 3.31911021e-07, 1.00264302e-01, 7.87608931e-03, 1.97721273e-03, 3.63975117e-01]

esumIpc15 = [16, 0.99448, 1.8715, 0.83188]
esumEpc15 = [15.206, 0.68424, 1.7012]
esumIpc18 = [17.953, 0.54388, 1.1256, 0.47898]
esumEpc18 = [17.410, 1.4282, 1.3006]
esumX17   = [18, 1.15]
esumX17   = [18.15, 1.15]

rangIpc15 = [1, -0.0218731, 0.00017231, -0.000000457066]
rangEpc15 = [35.316, 10.935, 10.935, 0.010000, 0.14216]
rangIpc18 = [1, -0.0218731, 0.00017231, -0.000000457066]
rangEpc18 = [28.236, 11.344, 11.344, 0.01, 0.17391]
rangX17   = [1.34020341e+02, 6.37571189e+00, 2.97597767e-03, 5.70417964e-05, 1.30502079e-01, 4.44284290e-03, 8.97301610e-03, 1.43534884e+02, 1.52415970e+01, 1.45209907e+00, 1.59235658e-03, 2.81801233e+00, 7.28489771e-02, 7.29979503e-05, 0.5487408362225106]
rangX17   = [136.39567, 9.5]
rangX17   = [138.92106967187416, 9.5]

# Definition of PDFs and likelihood functions
# 15 MeV IPC PDF parametrization
def integrate(func, args = (), xmin=0, xmax=1, n=100000):
    x = np.linspace(xmin, xmax, n)
    y = func(x, *args)

    dx = x[1] - x[0]
    y = y[1:] + y[:-1]
    y *= 0.5
    return dx*y.sum()

def tail(x, a, b):
    return np.exp(x*a + x**2*b)

def tail18(x, a, b, c):
    return np.exp(a/x + x*b + x**2*c)

def shoulder(x, x0, s1, l1, g1, s2, l2, g2):
    # Left side of the peak
    should = (x < x0)*np.exp(-(x - x0)**2/(2*(s1**2 + l1*x + g1*x**2)))
    # Right side of the peak
    should += (x > x0)*np.exp(-(x - x0)**2/(2*(s2**2 + l2*x + g2*x**2)))
    return should

def mIPC15(x, a, b, x0, s1, l1, g1, s2, l2, g2, F):
    # Normalize tail
    t = integrate(tail, xmin = mIPC15min, xmax = mIPC15max, args = (a,b))
    t = tail(x, a, b)/t

    # Normalize shoulder
    s = integrate(shoulder, xmin = mIPC15min, xmax = mIPC15max, args = (x0, s1, l1, g1, s2, l2, g2))
    s = shoulder(x, x0, s1, l1, g1, s2, l2, g2)/s

    return (1 - F)*t + F*s

def tIPC15(x, p0, p1, p2, p3):
    result = p0
    result += p1*x
    result += p2*x**2
    result += p3*x**3

    return result

def eIPC15(x, mu, sigma1, sigma2, f):
    result = f*np.exp(-0.5*(x - mu)**2/sigma1**2)/np.sqrt(2*np.pi)/sigma1
    result += (1 - f)*np.exp(-0.5*(x - mu)**2/sigma2**2)/np.sqrt(2*np.pi)/sigma2

    return result

def tEPC15(x, mu, sigmaL, sigmaR, alphaL, alphaR):
    result = (x < mu)*np.exp(-0.5*(x - mu)**2/(sigmaL**2 + (alphaL*(x - mu))**2))
    result += (x >= mu)*np.exp(-0.5*(x - mu)**2/(sigmaR**2 + (alphaR*(x - mu))**2))

    return result

def eEPC15(x, mu, sigmaL, sigmaR):
    result = (x < mu)*np.exp(-0.5*(x - mu)**2/sigmaL**2)
    result += (x >= mu)*np.exp(-0.5*(x - mu)**2/sigmaR**2)

    return result

def mIPC18(x, a, b, c, x0, s1, l1, g1, s2, l2, g2, F):
    # Normalize tail
    t = integrate(tail18, xmin = mIPC18min, xmax = mIPC18max, args = (a,b, c))
    t = tail18(x, a, b, c)/t

    # Normalize shoulder
    s = integrate(shoulder, xmin = mIPC18min, xmax = mIPC18max, args = (x0, s1, l1, g1, s2, l2, g2))
    s = shoulder(x, x0, s1, l1, g1, s2, l2, g2)/s

    return (1 - F)*t + F*s

def tIPC18(x, p0, p1, p2, p3):
    result = p0
    result += p1*x
    result += p2*x**2
    result += p3*x**3

    return result

def eIPC18(x, mu, sigma1, sigma2, f):
    result = f*np.exp(-0.5*(x - mu)**2/sigma1**2)/np.sqrt(2*np.pi)/sigma1
    result += (1 - f)*np.exp(-0.5*(x - mu)**2/sigma2**2)/np.sqrt(2*np.pi)/sigma2

    return result

def tEPC18(x, mu, sigmaL, sigmaR, alphaL, alphaR):
    result = (x < mu)*np.exp(-0.5*(x - mu)**2/(sigmaL**2 + (alphaL*(x - mu))**2))
    result += (x >= mu)*np.exp(-0.5*(x - mu)**2/(sigmaR**2 + (alphaR*(x - mu))**2))

    return result

def eEPC18(x, mu, sigmaL, sigmaR):
    result = (x < mu)*np.exp(-0.5*(x - mu)**2/sigmaL**2)
    result += (x >= mu)*np.exp(-0.5*(x - mu)**2/sigmaR**2)

    return result

def mX17(x, x0, s1, l1, g1, s2, l2, g2, x01, s11, l11, g11, s21, l21, g21, F):
    # Normalize shoulder
    s = integrate(shoulder, xmin = mX17min, xmax = mX17max, args = (x0, s1, l1, g1, s2, l2, g2))
    s = shoulder(x, x0, s1, l1, g1, s2, l2, g2)/s

    # Normalize shoulder
    S1 = integrate(shoulder, xmin = mX17min, xmax = mX17max, args = (x01, s11, l11, g11, s21, l21, g21))
    S1 = shoulder(x, x01, s11, l11, g11, s21, l21, g21)/S1

    return s*F + (1 - F)*S1

#def tX17(x, x0, s1, l1, g1, s2, l2, g2, x01, s11, l11, g11, s21, l21, g21, F):
#    # Normalize shoulder
#    s = integrate(shoulder, xmin = tX17min, xmax = tX17max, args = (x0, s1, l1, g1, s2, l2, g2))
#    s = shoulder(x, x0, s1, l1, g1, s2, l2, g2)/s
#
#    # Normalize shoulder
#    S1 = integrate(shoulder, xmin = tX17min, xmax = tX17max, args = (x01, s11, l11, g11, s21, l21, g21))
#    S1 = shoulder(x, x01, s11, l11, g11, s21, l21, g21)/S1
#
#    return s*F + (1 - F)*S1

def tX17(x, mu, sigma):
    result = np.exp(-0.5*(x - mu)**2/sigma**2)

    return result

def eX17(x, mu, sigma):
    return np.exp(-0.5*(x - mu)**2/sigma**2)

def sampleMassArray(_Nbkg = 400000, _fIPC18 = 0.25, _fIPC15 = 0.14, _fEPC18 = 0.48, _Nx17 = 500, year = 2021):
    # Sample number of events
    Nipc15 = np.random.poisson(_Nbkg*_fIPC15)
    Nipc18 = np.random.poisson(_Nbkg*_fIPC18)
    Nepc15 = np.random.poisson(_Nbkg*(1 - _fIPC15 - _fIPC18 - _fEPC18))
    Nepc18 = np.random.poisson(_Nbkg*_fEPC18)
    Nx17   = np.random.poisson(_Nx17)

    # Build imas cumulative
    m = np.linspace(12, 20, 100000)

    ipc15 = mIPC15(m, *imasIpc15)
    ipc18 = mIPC18(m, *imasIpc18)
    x17   = mX17(m, *imasX17)

    c_ipc15 = np.cumsum(ipc15)
    c_ipc15 -= c_ipc15.min()
    c_ipc15 /= c_ipc15.max()

    c_ipc18 = np.cumsum(ipc18)
    c_ipc18 -= c_ipc18.min()
    c_ipc18 /= c_ipc18.max()

    c_x17 = np.cumsum(x17)
    c_x17 -= c_x17.min()
    c_x17 /= c_x17.max()

    # Sample uniform
    s_ipc15 = np.random.uniform(0, 1, Nipc15 + Nepc15)
    f = interp1d(c_ipc15, m, kind='linear')
    m_ipc15 = f(s_ipc15)

    s_ipc18 = np.random.uniform(0, 1, Nipc18 + Nepc18)
    f = interp1d(c_ipc18, m, kind='linear')
    m_ipc18 = f(s_ipc18)
    
    s_x17 = np.random.uniform(0, 1, Nx17)
    f = interp1d(c_x17, m, kind='linear')
    m_x17 = f(s_x17)

    return m, c_ipc15, c_ipc18, c_x17

def sampleMass(_Nbkg = 400000, _fIPC18 = 0.25, _fIPC15 = 0.14, _fEPC18 = 0.48, _Nx17 = 500, year = 2021, SEED = 0, workDir = './', fileName=''):
    np.random.seed(SEED)
    # Sample number of events
    Nipc15 = np.random.poisson(_Nbkg*_fIPC15)
    Nipc18 = np.random.poisson(_Nbkg*_fIPC18)
    Nepc15 = np.random.poisson(_Nbkg*(1 - _fIPC15 - _fIPC18 - _fEPC18))
    Nepc18 = np.random.poisson(_Nbkg*_fEPC18)
    Nx17   = np.random.poisson(_Nx17)

    # Build imas cumulative
    m = np.linspace(12, 20, 100000)

    ipc15 = mIPC15(m, *imasIpc15)
    ipc18 = mIPC18(m, *imasIpc18)
    x17   = mX17(m, *imasX17)

    c_ipc15 = np.cumsum(ipc15)
    c_ipc15 -= c_ipc15.min()
    c_ipc15 /= c_ipc15.max()

    c_ipc18 = np.cumsum(ipc18)
    c_ipc18 -= c_ipc18.min()
    c_ipc18 /= c_ipc18.max()

    c_x17 = np.cumsum(x17)
    c_x17 -= c_x17.min()
    c_x17 /= c_x17.max()

    # Sample uniform
    s_ipc15 = np.random.uniform(0, 1, Nipc15 + Nepc15)
    f = interp1d(c_ipc15, m, kind='linear')
    m_ipc15 = f(s_ipc15)

    s_ipc18 = np.random.uniform(0, 1, Nipc18 + Nepc18)
    f = interp1d(c_ipc18, m, kind='linear')
    m_ipc18 = f(s_ipc18)
    
    s_x17 = np.random.uniform(0, 1, Nx17)
    f = interp1d(c_x17, m, kind='linear')
    m_x17 = f(s_x17)

    m_tot = m_ipc15
    m_tot = np.concatenate((m_tot, m_ipc18))
    m_tot = np.concatenate((m_tot, m_x17))

    # Build rang cumulative
    m = np.linspace(20, 180, 100000)

    ipc15 = tIPC15(m, *rangIpc15)
    mipc15 = m[ipc15 > 0]
    ipc15 = ipc15[ipc15 > 0]
    ipc18 = tIPC18(m, *rangIpc18)
    mipc18 = m[ipc18 > 0]
    ipc18 = ipc18[ipc18 > 0]
    epc15 = tEPC15(m, *rangEpc15)
    epc18 = tEPC18(m, *rangEpc18)
    x17   = tX17(m, *rangX17)
    #plt.plot(mipc15, ipc15)
    #plt.show()
    #plt.plot(mipc15, ipc18)
    #plt.show()
    #plt.plot(m, epc15)
    #plt.show()
    #plt.plot(m, epc18)
    #plt.show()
    #plt.plot(m, x17)
    #plt.show()

    c_ipc15 = np.cumsum(ipc15)
    c_ipc15 -= c_ipc15.min()
    c_ipc15 /= c_ipc15.max()

    c_ipc18 = np.cumsum(ipc18)
    c_ipc18 -= c_ipc18.min()
    c_ipc18 /= c_ipc18.max()

    c_epc15 = np.cumsum(epc15)
    c_epc15 -= c_epc15.min()
    c_epc15 /= c_epc15.max()

    c_epc18 = np.cumsum(epc18)
    c_epc18 -= c_epc18.min()
    c_epc18 /= c_epc18.max()

    c_x17 = np.cumsum(x17)
    c_x17 -= c_x17.min()
    c_x17 /= c_x17.max()

    # Do relative angle
    s_ipc15 = np.random.uniform(0, 1, Nipc15)
    s_ipc18 = np.random.uniform(0, 1, Nipc18)
    s_epc15 = np.random.uniform(0, 1, Nepc15)
    s_epc18 = np.random.uniform(0, 1, Nepc18)
    s_x17   = np.random.uniform(0, 1, Nx17)

    f = interp1d(c_ipc15, mipc15, kind='linear')
    t_ipc15 = f(s_ipc15)
    #plt.hist(t_ipc15, bins=50)
    #plt.show()
    #plt.plot(mipc15, c_ipc15)
    #plt.show()

    f = interp1d(c_ipc18, mipc18, kind='linear')
    t_ipc18 = f(s_ipc18)
    #plt.hist(t_ipc18, bins=50)
    #plt.show()
    #plt.plot(mipc18, c_ipc18)
    #plt.show()
    
    f = interp1d(c_epc15, m, kind='linear')
    t_epc15 = f(s_epc15)
    #plt.hist(t_epc15, bins=50)
    #plt.show()
    #plt.plot(m, c_epc15)
    #plt.show()

    f = interp1d(c_epc18, m, kind='linear')
    t_epc18 = f(s_epc18)
    #plt.hist(t_epc18, bins=50)
    #plt.show()
    #plt.plot(m, c_epc18)
    #plt.show()
    
    f = interp1d(c_x17, m, kind='linear')
    t_x17 = f(s_x17)
    #plt.hist(t_x17, bins=50)
    #plt.show()
    
    t_tot = t_ipc15
    t_tot = np.concatenate((t_tot, t_ipc18))
    t_tot = np.concatenate((t_tot, t_epc15))
    t_tot = np.concatenate((t_tot, t_epc18))
    t_tot = np.concatenate((t_tot, t_x17))

    # Build esum cumulative
    m = np.linspace(10, 24, 100000)
    s_ipc15 = np.random.uniform(0, 1, Nipc15)
    s_ipc18 = np.random.uniform(0, 1, Nipc18)
    s_epc15 = np.random.uniform(0, 1, Nepc15)
    s_epc18 = np.random.uniform(0, 1, Nepc18)
    s_x17   = np.random.uniform(0, 1, Nx17)

    ipc15 = eIPC15(m, *esumIpc15)
    #plt.plot(m, ipc15)
    #plt.show()
    ipc18 = eIPC18(m, *esumIpc18)
    epc15 = eEPC15(m, *esumEpc15)
    epc18 = eEPC18(m, *esumEpc18)
    x17   = eX17(m, *esumX17)
    print(eX17(17, *esumX17),eEPC15(17, *esumEpc15),eIPC15(17, *esumIpc15),eEPC18(17, *esumEpc18),eIPC18(17, *esumIpc18))
    #plt.plot(m, x17)
    #plt.show()

    #print(ipc15)
    c_ipc15 = np.cumsum(ipc15)
    c_ipc15 -= c_ipc15.min()
    c_ipc15 /= c_ipc15.max()

    c_ipc18 = np.cumsum(ipc18)
    c_ipc18 -= c_ipc18.min()
    c_ipc18 /= c_ipc18.max()

    c_epc15 = np.cumsum(epc15)
    c_epc15 -= c_epc15.min()
    c_epc15 /= c_epc15.max()

    c_epc18 = np.cumsum(epc18)
    c_epc18 -= c_epc18.min()
    c_epc18 /= c_epc18.max()

    c_x17 = np.cumsum(x17)
    c_x17 -= c_x17.min()
    c_x17 /= c_x17.max()

    # Do relative angle
    f = interp1d(c_ipc15, m, kind='linear')
    e_ipc15 = f(s_ipc15)

    f = interp1d(c_ipc18, m, kind='linear')
    e_ipc18 = f(s_ipc18)
    
    f = interp1d(c_epc15, m, kind='linear')
    e_epc15 = f(s_epc15)

    f = interp1d(c_epc18, m, kind='linear')
    e_epc18 = f(s_epc18)
    
    f = interp1d(c_x17, m, kind='linear')
    e_x17 = f(s_x17)
    
    e_tot = e_ipc15
    e_tot = np.concatenate((e_tot, e_ipc18))
    e_tot = np.concatenate((e_tot, e_epc15))
    e_tot = np.concatenate((e_tot, e_epc18))
    e_tot = np.concatenate((e_tot, e_x17))

    # Write to file
    if fileName == '':
        prefix = 'X17MC%d_s%d.root' %(year, SEED)
    else:
        prefix = fileName + '_s%d.root' %(SEED)
    with uproot.recreate(workDir + prefix) as file:
        run = np.zeros(len(m_tot)) + year*1000
        event = np.linspace(0, len(m_tot)-1, len(m_tot))
        year = np.zeros(len(m_tot)) + year
        ecode = np.zeros(len(e_ipc15)) + 2
        ecode = np.concatenate((ecode, np.zeros(len(e_ipc18)) + 4))
        ecode = np.concatenate((ecode, np.zeros(len(e_epc15)) + 1))
        ecode = np.concatenate((ecode, np.zeros(len(e_epc18)) + 3))
        ecode = np.concatenate((ecode, np.zeros(len(e_x17))))
        cat = np.ones(len(m_tot))
        #run = run[t_tot > 100]
        #event = event[t_tot > 100]
        #year = year[t_tot > 100]
        #ecode = ecode[t_tot > 100]
        #cat = cat[t_tot > 100]
        #m_tot = m_tot[t_tot > 100]
        #e_tot = e_tot[t_tot > 100]
        #t_tot = t_tot[t_tot > 100]
        file['ntuple'] = {'run' : run.astype('float32'), 'event' : event.astype('float32'), 'year' : year.astype('float32'), 'ecode' : ecode.astype('float32'), 'esum' : e_tot.astype('float32'), 'imas' : m_tot.astype('float32'), 'dth' : t_tot.astype('float32'), 'cat' : cat.astype('float32')}

if __name__ ==  '__main__':
    sampleMass(_Nbkg = 250000, _fIPC18 = 0.20, _fIPC15 = 0.11, _fEPC18 = 0.54, _Nx17 = 450, year = 2021)
    sampleMass(_Nbkg = 250000, _fIPC18 = 0.20, _fIPC15 = 0.11, _fEPC18 = 0.54, _Nx17 = 0, year = 2021)
    sampleMass(_Nbkg = 400000, _fIPC18 = 0.25, _fIPC15 = 0.25, _fEPC18 = 0.25, _Nx17 = 100000, year = 2021, SEED=299792458)
    sampleMass(_Nbkg = 220000, _fIPC18 = 0.4545454545, _fIPC15 = 0.4545454545, _fEPC18 = 0.04545454545, _Nx17 = 100000, year = 2021, SEED = 299792459, workDir = '')
    for i in range(100):
        sampleMass(_Nbkg = 400000, _fIPC18 = 0.25, _fIPC15 = 0.25, _fEPC18 = 0.25, _Nx17 = 100000, year = 2021, SEED=299792458 + i) 
        #sampleMass(_Nbkg = 220000, _fIPC18 = 0.45454545454545453, _fIPC15 = 0.45454545454545453, _fEPC18 = 0.045454545454545453, _Nx17 = 100000, year = 2021, SEED=i + 299792458)
        refs/remotes/origin/main

