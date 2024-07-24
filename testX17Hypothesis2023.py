import X17pythonTask_2023
import numpy as np
import time
from matplotlib import pyplot as plt
import matplotlib
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nToys', type=int, default=100)
    parser.add_argument('--SEED', type=int, default=0)
    parser.add_argument('--SampleBranch', type=int, default=0)
    parser.add_argument('--nX17', type=float, default=16.97)
    parser.add_argument('--pX17', type=float, default=1)
    return parser.parse_args()

startTime = time.time()

massElectron = 0.5109989461 #MeV

ratio176 = 1.97 # +- 0.15
ratio179 = 0.93 # +- 0.07
ratio181 = 0.74 # +- 0.06

dRatio176 = 0.15
dRatio179 = 0.07
dRatio181 = 0.06

p176 = ratio176/(ratio176 + 1)
dP176 = dRatio176/(ratio176 + 1)**2
p179 = ratio179/(ratio179 + 1)
dP179 = dRatio179/(ratio179 + 1)**2
p181 = ratio181/(ratio181 + 1)
dP181 = dRatio181/(ratio181 + 1)**2

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

esumCutLow = 20   # No cut
esumCutHigh = 15  # No cut
angleCutLow = 180 # No cut
angleCutHigh = 0  # No cut

dataRunMax = 511000

workDir = './results/'
#workDir = '/data/project/general/muonGroup/simulations/giovanni/X17BBPythonTask/results/'
dataFile = 'data2023.root'
MCFile = 'MC2023tot.root'

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
    'ch4': {
        'name': 'X17 2023, high energy, high angle',
        'Esum': [16, 20, 6],
        'Angle': [80, 180, 10]
    },
}

BKGnames = ['IPC 17.6', 'IPC 17.9', 'IPC 18.1', 'IPC 14.6', 'IPC 14.9', 'IPC 15.1', 'EPC 18', 'EPC 15', 'Fakes']
alphaNames = ['res', 'field']

if __name__ == '__main__':
    args = parse_args()
    nToys = args.nToys
    SEED = args.SEED
    SampleBranch = bool(args.SampleBranch)
    nX17 = args.nX17
    pX17 = args.pX17
    
    
    startingPars = np.array([100, 0.5, 16.8, 4e5, 0, 1e4, p176, p179, p181, 1, 1e4, 1e4, 0, 0, 0])
    FixedParameters = np.array([False, False, False, False, False, False, False, False, False, False, False, False, False, True, False])

    ################################################################################################################################################
    # Positive signal hypothesis fit
    print('Starting positive signal hypothesis fit')
    scalingFactor = [1/3., 1/3., 1/3., 1/3., 1/3., 1/3., 1., 1., 1,]
    angleUS = 50
    negSignal = False

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
    # Fit

    nullHypothesis = False

    logL, betas, MAXLikelihood = X17pythonTask_2023.bestFit(startingPars, Hists, FitToy = False, doNullHypothesis = nullHypothesis, FixedParameters = FixedParameters, negSignal = negSignal)

    BestBetas = np.copy(betas)
    BestPars = np.copy(logL.values)
    BestErrors = np.copy(logL.errors)

    print('Best fit values: ', BestPars)
    print('Correlation matrix: ', logL.covariance.correlation())

    PARS = []
    Likelihood = []
    Accurate = []
    Valid = []
    totPars = logL.values
    BETAS = betas

    print('Time elapsed: ', time.time() - startTime)


    # Test X17 hypothesis
    SignalYield = nX17
    SignalFraction = pX17
    SignalMass = 16.97
    FixedParameters[1] = True
    X17pythonTask_2023.testHypothesis(SignalYield, SignalFraction, SignalMass, logL, Hists, BestPars, SEED = SEED, nToys = nToys, betas = BestBetas, nus = 1, fluctuateTemplates = True, FixedParameters = FixedParameters, PARS = [], Likelihood = [], Accurate = [], Valid = [], Toy = [], workDir = workDir, doingDataToy = False, negSignal = False, oldMass = 16.97, constrainMass = True, oldP176 = p176, oldP179 = p179, oldP181 = p181, oldAlphaField = 0, SampleBranching=SampleBranch)
    print('Time elapsed: ', time.time() - startTime)
