import X17pythonTask_2023
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib

startTime = time.time()

massElectron = 0.5109989461 #MeV

#ratio176 = 3.84 # +- 0.27
ratio176 = 1.97 # +- 0.15
ratio179 = 0.93 # +- 0.07
ratio181 = 0.74 # +- 0.06

#dRatio176 = 0.27
dRatio176 = 0.15
dRatio179 = 0.07
dRatio181 = 0.06

p176 = ratio176/(ratio176 + 1)
dP176 = dRatio176/(ratio176 + 1)**2
p179 = ratio179/(ratio179 + 1)
dP179 = dRatio179/(ratio179 + 1)**2
p181 = ratio181/(ratio181 + 1)
dP181 = dRatio181/(ratio181 + 1)**2

# Nuisances
# Sampled PDF centers
_p176 = p176
_p179 = p179
_p181 = p181
_alphaField = 0

# Best fit values
newP176 = p176
newP179 = p179
newP181 = p181
newAlphaField = 0

# Field scale Fomr Hicham
OldScale = 0.152
NewScale = 0.1537
dNewScale = 0.0002

Correction = OldScale/NewScale
dCorrection = dNewScale*OldScale/NewScale**2

dAlphaField = dCorrection

# Best Likelihood
MAXLikelihood = 0

alphares = 0.005
alphafield = 0.005

esumCutLow = 16
esumCutHigh = 20
angleCutLow = 115
angleCutHigh = 160

#esumCutLow = 20
#esumCutHigh = 15
#angleCutLow = 180
#angleCutHigh = 0

dataRunMax = 511000

workDir = './results/'
dataFile = 'data2023.root'
MCFile = 'MC2023tot.root'
#MCFile = 'MC2023tot_initialX17statistics.root'

ECODETYPE = 'ecode'

X17masses = np.array([16.3, 16.5, 16.7, 16.9, 17.1, 17.3])
dX17mass = 0.001
massRef = 16.9
fitFakes = False

channels = {
    'ch1': {
        'name': 'X17 2023, low angle, low energy',
        'Esum': [15, 16, 2], # [min, max, nBins]
        'Angle': [0, 80, 16]
    },
    'ch2': {
        'name': 'X17 2023, low angle, high energy',
        'Esum': [16, 20, 8], # [min, max, nBins]
        'Angle': [30, 80, 10]
    },
    'ch3': {
        'name': 'X17 2023, low energy, high angle',
        'Esum': [15, 16, 1],
        'Angle': [80, 180, 5]
    },
    #'ch4': {
    #    'name': 'X17 2023, low energy, high angle, last bin',
    #    'Esum': [15, 16, 1],
    #    'Angle': [160, 170, 1]
    #},
    'ch5': {
        'name': 'X17 2023, high energy, high angle',
        'Esum': [16, 20, 6],
        'Angle': [80, 180, 10]
    },
}


channelsUp = {
    'ch1': {
        'name': 'X17 2023, low angle, low energy',
        'Esum': [15, 16, 4], # [min, max, nBins]
        'Angle': [0, 80, 32]
    },
    'ch2': {
        'name': 'X17 2023, low angle, high energy',
        'Esum': [16, 20, 16], # [min, max, nBins]
        'Angle': [30, 80, 20]
    },
    'ch3': {
        'name': 'X17 2023, low energy, high angle',
        'Esum': [15, 16, 2],
        'Angle': [80, 180, 10]
    },
    #'ch4': {
    #    'name': 'X17 2023, low energy, high angle, last bin',
    #    'Esum': [15, 16, 1],
    #    'Angle': [160, 170, 1]
    #},
    'ch5': {
        'name': 'X17 2023, high energy, high angle',
        'Esum': [16, 20, 4],
        'Angle': [80, 180, 20]
    },
}

#channels = channelsUp

BKGnames = ['IPC 17.6', 'IPC 17.9', 'IPC 18.1', 'IPC 14.6', 'IPC 14.9', 'IPC 15.1', 'EPC 18', 'EPC 15', 'Fakes']

scalingFactor = [1/2., 1/2., 1/2., 1/2., 1/2., 1/2., 1., 1., 1,]
scalingFactor = [1/3., 1/3., 1/3., 1/3., 1/3., 1/3., 1., 1., 1,]
#scalingFactor = 1

alphaNames = ['res', 'field']

################################################################################################################################################
angleUS = 50

# Load data
TotalDataNumber, channels = X17pythonTask_2023.readData(channels, workDir = workDir, dataFile = dataFile, dataRunMax = dataRunMax, angleUS = angleUS)
print('TotalDataNumber: ', TotalDataNumber)

X17pythonTask_2023.plotChannels(channels, sample='dataHist', title='Data')

# Load MC
simbeamEnergy = {
    'IPC400': [0.42, 0.46], # energy in MeV of simulated proton beam
    'IPC700': [0.593, 0.836], # energy in MeV of simulated proton beam
    'IPC1000': [0.980, 1.060] # energy in MeV of simulated proton beam
    }

TotalMCStatistics, nBKGs, channels = X17pythonTask_2023.readMC(channels, CUTfile = workDir + 'MC2023totOLDmerge.root:ntuple', workDir = workDir, MCFile = MCFile, ECODETYPE = ECODETYPE, X17masses = X17masses, dX17mass = dX17mass, alphares = alphares, alphafield = alphafield, esumCutLow = esumCutLow, esumCutHigh = esumCutHigh, angleCutLow = angleCutLow, angleCutHigh = angleCutHigh, BKGnames = BKGnames, alphaNames = alphaNames, scalingFactor = scalingFactor, simbeamEnergy = simbeamEnergy, angleUS = angleUS)

print('TotalMCStatistics: ', len(TotalMCStatistics))

alphavalues = [np.linspace(-5*alphares, 5*alphares, 11), np.linspace(-5*alphafield, 5*alphafield, 11)]
alphaRefs = [0, 0]
Hists = X17pythonTask_2023.histHandler(channels, 'dataHist', ['X17_17.6', 'X17_18.1'], BKGnames, 'Esum', 'Angle', alphaNames, alphavalues, alphaRefs, TotalMCStatistics=np.array(TotalMCStatistics), masses=X17masses, massRef=massRef)

print(Hists.BKGarrayNuisance5SigmaArray.shape)


################################################################################################################################################
# Fit sidebands
startingPars = np.array([100, 0.5,  16.9, 4e5, 1e4, 1e4, p176, p179, p181, 1, 1e4, 1e4, 0, 0, 0])
FixedParameters = np.array([False, False, False, False, False, False, False, False, False, False, False, False, False, True, False])

nullHypothesis = True

logL, betas, MAXLikelihood = X17pythonTask_2023.bestFit(startingPars, Hists, FitToy = False, doNullHypothesis = nullHypothesis, FixedParameters = FixedParameters)

BestBetas = np.copy(betas)
BestPars = np.copy(logL.values)

print('Best fit values: ', BestPars)
print('Correlation matrix: ', logL.covariance.correlation())

#X17pythonTask_2023.plotComparison(Hists, logL.values, betas, channels, compareWithBetas=False, logL = logL, BKGnames = BKGnames, CHANNEL='ch5', LOGARITMIC=True)

PARS = []
Likelihood = []
Accurate = []
Valid = []
totPars = logL.values
BETAS = betas

#PARS, Likelihood, Accurate, Valid = X17pythonTask_2023.GoodnessOfFit(logL, Hists, BestBetas, BestPars, channels, nToys = 1000, doNullHypothesis = nullHypothesis, FixedParameters = FixedParameters, PARS = PARS, Likelihood = Likelihood, Accurate = Accurate, Valid = Valid)

print('Time elapsed: ', time.time() - startTime)


################################################################################################################################################
# FC test
esumCutLow = 20
esumCutHigh = 15
angleCutLow = 180
angleCutHigh = 0
TotalMCStatistics = []

SignalYield = 254
SignalFraction = 90./254
SignalMass = 16.97
print('Betas: ', BestBetas)
#BestBetas = 1

#FixedParameters[1] = True
#FixedParameters[2] = True

startingPars[0] = SignalYield
startingPars[1] = SignalFraction
startingPars[2] = SignalMass

TotalMCStatistics, nBKGs, channelsTest = X17pythonTask_2023.readMC(channels, CUTfile = workDir + 'MC2023totOLDmerge.root:ntuple', workDir = workDir, MCFile = MCFile, ECODETYPE = ECODETYPE, X17masses = X17masses, dX17mass = dX17mass, alphares = alphares, alphafield = alphafield, esumCutLow = esumCutLow, esumCutHigh = esumCutHigh, angleCutLow = angleCutLow, angleCutHigh = angleCutHigh, BKGnames = BKGnames, alphaNames = alphaNames)

HistsTest = X17pythonTask_2023.histHandler(channelsTest, 'dataHist', ['X17_17.6', 'X17_18.1'], BKGnames, 'Esum', 'Angle', alphaNames, alphavalues, alphaRefs, TotalMCStatistics=np.array(TotalMCStatistics), masses=X17masses, massRef=massRef)

yields = np.concatenate([X17pythonTask_2023.getSignalYields(SignalYield, SignalFraction), X17pythonTask_2023.getYields(BestPars[3], BestPars[4], BestPars[5], BestPars[6], BestPars[7], BestPars[8], BestPars[2]), [BestPars[10], BestPars[11], BestPars[12]]])


# Do Toy
esumCutLow = 20
esumCutHigh = 15
angleCutLow = 180
angleCutHigh = 0
TotalMCStatistics = []

np.random.seed(0)
TotalMCStatistics, nBKGs, channels = X17pythonTask_2023.readMC(channels, CUTfile = workDir + 'MC2023totOLDmerge.root:ntuple', workDir = workDir, MCFile = MCFile, ECODETYPE = ECODETYPE, X17masses = X17masses, dX17mass = dX17mass, alphares = alphares, alphafield = alphafield, esumCutLow = esumCutLow, esumCutHigh = esumCutHigh, angleCutLow = angleCutLow, angleCutHigh = angleCutHigh, BKGnames = BKGnames, alphaNames = alphaNames, scalingFactor = scalingFactor, simbeamEnergy = simbeamEnergy)

HistsTest = X17pythonTask_2023.histHandler(channels, 'dataHist', ['X17_17.6', 'X17_18.1'], BKGnames, 'Esum', 'Angle', alphaNames, alphavalues, alphaRefs, TotalMCStatistics=np.array(TotalMCStatistics), masses=X17masses, massRef=massRef)
HistsTest.generateToy(yields, betas = BestBetas, fluctuateTemplates = True, morph = BestPars[-2:], mass=SignalMass)

HistsTest.DataArray = np.copy(HistsTest.DataArrayToy)
#
HistsTest.SignalArray = np.copy(HistsTest.SignalArrayToy)
HistsTest.SignalArrayNuisance = np.copy(HistsTest.SignalArrayNuisanceToy)
HistsTest.SignalArrayNuisance5Sigma = np.copy(HistsTest.SignalArrayNuisance5SigmaToy)
HistsTest.SignalArrayNuisance5SigmaArray = np.copy(HistsTest.SignalArrayNuisance5SigmaArrayToy)

HistsTest.BKGarray = np.copy(HistsTest.BKGarrayToy)
HistsTest.BKGarrayNuisance5Sigma = np.copy(HistsTest.BKGarrayNuisance5SigmaToy)
HistsTest.BKGarrayNuisance5SigmaArray = np.copy(HistsTest.BKGarrayNuisance5SigmaArrayToy)

startingPars = np.array(logL.values)
startingPars[:3] = [SignalYield, SignalFraction, SignalMass]
print('Starting pars: ', startingPars)


logL, betas, MAXLikelihood = X17pythonTask_2023.bestFit(startingPars, HistsTest, FitToy = False, doNullHypothesis = True, FixedParameters = FixedParameters, _p176 = _p176, _p179 = _p179, _p181 = _p181, _alphaField = _alphaField)

toyBestPars = np.copy(logL.values)
toyBestBetas = np.copy(betas)
X17pythonTask_2023.plotComparison(HistsTest, logL.values, toyBestBetas, channels, compareWithBetas=False, logL = logL, BKGnames = BKGnames, CHANNEL='ch5', LOGARITMIC=True, TITLE='')

logL, betas, MAXLikelihood = X17pythonTask_2023.bestFit(startingPars, HistsTest, FitToy = False, doNullHypothesis = False, FixedParameters = FixedParameters, _p176 = _p176, _p179 = _p179, _p181 = _p181, _alphaField = _alphaField)

toyBestPars = np.copy(logL.values)
toyBestBetas = np.copy(betas)

testPars = ['nSig', 'pSig181', 'mass']
#for name in testPars:
#    matplotlib.rcParams.update({'font.size': 30})
#    fig = plt.figure(figsize=(21, 14), dpi=100)
#    #if name == 'pSig181':
#    #    logL.draw_profile(name, bound = [0,1])
#    #    logL.draw_mnprofile(name, bound = [0,1])
#    #else:
#    #    logL.draw_profile(name)
#    #    logL.draw_mnprofile(name)
#    logL.draw_profile(name)
#    logL.draw_mnprofile(name)
#    plt.axhline(1, color='red')
#    plt.ylim(0, plt.gca().get_ylim()[1])

toyBestPars = np.copy(logL.values)
X17pythonTask_2023.plotComparison(HistsTest, logL.values, toyBestBetas, channels, compareWithBetas=False, logL = logL, BKGnames = BKGnames, CHANNEL='ch5', LOGARITMIC=True, TITLE='')
logL.values[0] = 0
logL.covariance[0,0] = 0
logL.covariance[1,1] = 0
logL.covariance[2,2] = 0
X17pythonTask_2023.plotComparison(HistsTest, logL.values, toyBestBetas, channels, compareWithBetas=False, logL = logL, BKGnames = BKGnames, CHANNEL='ch5', LOGARITMIC=True, TITLE='')


toyBestPars = np.array([0, 0.5, 16.9, 1.2e5, 0, 8e4, p176, p179, p181, 1, 2.2e5, 1.1e5, 1300, 0, 0])
toyBestBetas = 1

toyBestPars[0] = SignalYield
toyBestPars[1] = SignalFraction
toyBestPars[2] = SignalMass

toyBestPars[0] = 0
toyBestPars[1] = SignalFraction
toyBestPars[2] = SignalMass

PARS = []
Likelihood = []
Accurate = []
Valid = []

#PARS, Likelihood, Accurate, Valid = X17pythonTask_2023.GoodnessOfFit(logL, HistsTest, toyBestBetas, toyBestPars, channels, nToys = 1000, doNullHypothesis = False, FixedParameters = FixedParameters, PARS = PARS, Likelihood = Likelihood, Accurate = Accurate, Valid = Valid)


## Signal grid point
#SignalYield = logL.values[0]
#SignalMass = logL.values[1]
#SignalYield = 100
#SignalFraction = 70./270
#SignalMass = 16.9
#
#PARS, Likelihood, Accurate, Valid, Toy, FixedParameters = X17pythonTask_2023.FCgenerator(SignalYield, SignalFraction, SignalMass, logL, HistsTest, toyBestPars, SEED = 0, nToys = 10, betas = toyBestBetas, fluctuateTemplates=True, FixedParameters = FixedParameters, PARS = [], Likelihood = [], Accurate = [], Valid = [], Toy = [], workDir = workDir)

print('Time elapsed: ', time.time() - startTime)