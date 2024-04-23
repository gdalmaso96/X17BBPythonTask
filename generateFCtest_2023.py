import X17pythonTask_2023
import numpy as np
import time
import argparse

startTime = time.time()

#python3 generateFCtest_2023.py --SignalYield 500 --SignalMass 16.9 --SEED 0 --nToys 10

def argparser():
    parser = argparse.ArgumentParser(description='Generate FC test')
    parser.add_argument('--SignalYield', type=int, default=0, help='Signal yield')
    parser.add_argument('--SignalMass', type=float, default=16.9, help='Signal mass')
    parser.add_argument('--SEED', type=int, default=0, help='Seed')
    parser.add_argument('--nToys', type=int, default=10, help='Number of toys')
    return parser.parse_args()

massElectron = 0.5109989461 #MeV

ratio176 = 3.84 # +- 0.27
ratio179 = 0.93 # +- 0.07
ratio181 = 0.74 # +- 0.06

dRatio176 = 0.27
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
        'Angle': [80, 160, 4]
    },
    'ch4': {
        'name': 'X17 2023, low energy, high angle, last bin',
        'Esum': [15, 16, 1],
        'Angle': [160, 170, 1]
    },
    'ch5': {
        'name': 'X17 2023, high energy, high angle',
        'Esum': [16, 20, 2],
        'Angle': [80, 170, 9]
    },
}

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

ECODETYPE = 'ecode'

X17masses = np.array([16.3, 16.5, 16.7, 16.9, 17.1, 17.3])
dX17mass = 0.0001
massRef = 16.9

BKGnames = ['IPC 17.6', 'IPC 17.9', 'IPC 18.1', 'IPC 14.6', 'IPC 14.9', 'IPC 15.1', 'EPC 18', 'EPC 15']

alphaNames = ['res', 'field']

if __name__ == '__main__':
    args = argparser()
    SignalYield = args.SignalYield
    SignalMass = args.SignalMass
    SEED = args.SEED
    nToys = args.nToys
    
    ################################################################################################################################################
    # Load data
    TotalDataNumber, channels = X17pythonTask_2023.readData(channels, workDir = workDir, dataFile = dataFile, dataRunMax = dataRunMax)

    # Load MC
    TotalMCStatistics, nBKGs, channels = X17pythonTask_2023.readMC(channels, CUTfile = workDir + 'MC2023totOLDmerge.root:ntuple', workDir = workDir, MCFile = MCFile, ECODETYPE = ECODETYPE, X17masses = X17masses, dX17mass = dX17mass, alphares = alphares, alphafield = alphafield, esumCutLow = esumCutLow, esumCutHigh = esumCutHigh, angleCutLow = angleCutLow, angleCutHigh = angleCutHigh, BKGnames = BKGnames, alphaNames = alphaNames)

    alphavalues = [np.linspace(-5*alphares, 5*alphares, 11), np.linspace(-5*alphafield, 5*alphafield, 11)]
    alphaRefs = [0, 0]
    Hists = X17pythonTask_2023.histHandler(channels, 'dataHist', 'X17', BKGnames, 'Esum', 'Angle', alphaNames, alphavalues, alphaRefs, TotalMCStatistics=np.array(TotalMCStatistics), masses=X17masses, massRef=massRef)


    ################################################################################################################################################
    # Fit sidebands
    startingPars = np.array([100, 16.9, 4e5, 1e4, 1e4, p176, p179, p181, 1, 1e4, 1e4, 0, 0])
    FixedParameters = np.array([False, False, False, False, False, False, False, False, False, False, False, True, False])

    logL, betas, MAXLikelihood = X17pythonTask_2023.bestFit(startingPars, Hists, FitToy = False, doNullHypothesis = True, FixedParameters = FixedParameters)

    BestBetas = np.copy(betas)
    BestPars = np.copy(logL.values)
    newP176 = logL.values[5]
    newP179 = logL.values[6]
    newP181 = logL.values[7]
    newAlphaField = logL.values[12]
    _p176 = logL.values[5]
    _p179 = logL.values[6]
    _p181 = logL.values[7]
    _alphaField = logL.values[12]

    print('Time elapsed to prepare the parameters: ', time.time() - startTime)
    
    ################################################################################################################################################
    # FC test
    esumCutLow = 20
    esumCutHigh = 15
    angleCutLow = 180
    angleCutHigh = 0
    TotalMCStatistics = []

    TotalMCStatistics, nBKGs, channelsTest = X17pythonTask_2023.readMC(channels, CUTfile = workDir + 'MC2023totOLDmerge.root:ntuple', workDir = workDir, MCFile = MCFile, ECODETYPE = ECODETYPE, X17masses = X17masses, dX17mass = dX17mass, alphares = alphares, alphafield = alphafield, esumCutLow = esumCutLow, esumCutHigh = esumCutHigh, angleCutLow = angleCutLow, angleCutHigh = angleCutHigh, BKGnames = BKGnames, alphaNames = alphaNames)

    HistsTest = X17pythonTask_2023.histHandler(channelsTest, 'dataHist', 'X17', BKGnames, 'Esum', 'Angle', alphaNames, alphavalues, alphaRefs, TotalMCStatistics=np.array(TotalMCStatistics), masses=X17masses, massRef=massRef)
    
    FixedParameters = np.array([False, False, False, False, False, False, False, False, False, False, False, True, False])

    yields = np.concatenate([[BestPars[0]], X17pythonTask_2023.getYields(BestPars[2], BestPars[3], BestPars[4], BestPars[5], BestPars[6], BestPars[7], BestPars[8]), [BestPars[9], BestPars[10]]])

    np.random.seed(0)
    HistsTest.generateToy(yields, betas = 1, fluctuateTemplates = True, morph = BestPars[-2:], mass=BestPars[1])

    HistsTest.DataArray = np.copy(HistsTest.DataArrayToy)

    FixedParameters = np.array([False, False, False, False, False, False, False, False, False, False, False, True, False])
    
    print(BestPars)

    # Signal grid point
    print('Start FC generation after ', time.time() - startTime, ' seconds')
    PARS, Likelihood, Accurate, Valid, Toy, FixedParameters = X17pythonTask_2023.FCgenerator(SignalYield, SignalMass, logL, HistsTest, BestPars, SEED = SEED, nToys = nToys, betas = betas, fluctuateTemplates=True, FixedParameters = FixedParameters, PARS = [], Likelihood = [], Accurate = [], Valid = [], Toy = [], workDir = workDir)
    print('Time elapsed to generate FC: ', time.time() - startTime)