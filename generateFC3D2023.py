import X17pythonTask_2023
import numpy as np
import time
import argparse

startTime = time.time()

#python3 generateFC3Dtest_2023.py --SignalYield 500 --SignalFraction 0 --SignalMass 16.9 --SEED 0 --nToys 10
#python3 generateFC3Dtest_2023.py --SignalYield   0 --SignalFraction 0 --SignalMass 16.9 --SEED   0 --nToys 1 --generateDataSets True --DataSEED 0 --DataSignalMass 16.9 --DataSignalFraction 0 --DataSignalYield 0

def argparser():
    parser = argparse.ArgumentParser(description='Generate FC test')
    parser.add_argument('--SignalYield', type=int, default=0, help='Signal yield')
    parser.add_argument('--SignalFraction', type=float, default=0, help='Signal fraction')
    parser.add_argument('--SignalMass', type=float, default=16.9, help='Signal mass')
    parser.add_argument('--SEED', type=int, default=0, help='Seed')
    parser.add_argument('--nToys', type=int, default=10, help='Number of toys')
    parser.add_argument('--generateDataSets', type=bool, default=False, help='Sample data sets')
    parser.add_argument('--DataSEED', type=int, default=0, help='Data set seed')
    parser.add_argument('--DataSignalMass', type=float, default=16.9, help='Data set signal mass')
    parser.add_argument('--DataSignalFraction', type=float, default=0, help='Data set signal fraction')
    parser.add_argument('--DataSignalYield', type=int, default=0, help='Data set signal yield')
    return parser.parse_args()

massElectron = 0.5109989461 #MeV

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
        'Angle': [80, 180, 5]
    },
    'ch4': {
        'name': 'X17 2023, high energy, high angle',
        'Esum': [16, 20, 6],
        'Angle': [80, 180, 10]
    },
}

alphares = 0.005
alphafield = 0.005

esumCutLow = 16
esumCutHigh = 20
angleCutLow = 115
angleCutHigh = 160

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

BKGnames = ['IPC 17.6', 'IPC 17.9', 'IPC 18.1', 'IPC 14.6', 'IPC 14.9', 'IPC 15.1', 'EPC 18', 'EPC 15', 'Fakes']

alphaNames = ['res', 'field']

scalingFactor = [1/3., 1/3., 1/3., 1/3., 1/3., 1/3., 1., 1., 1.]

angleUS = 50

simbeamEnergy = {
    'IPC400': [0.42, 0.46], # energy in MeV of simulated proton beam
    'IPC700': [0, 2], # energy in MeV of simulated proton beam
    'IPC1000': [0.980, 1.060] # energy in MeV of simulated proton beam
    }


if __name__ == '__main__':
    args = argparser()
    SignalYield = args.SignalYield
    SignalFraction = args.SignalFraction
    SignalMass = args.SignalMass
    SEED = args.SEED
    nToys = args.nToys
    generateDataSets = args.generateDataSets
    DataSEED = args.DataSEED
    DataSignalMass = args.DataSignalMass
    DataSignalFraction = args.DataSignalFraction
    DataSignalYield = args.DataSignalYield
    
    ################################################################################################################################################
    # Load data
    TotalDataNumber, channels = X17pythonTask_2023.readData(channels, workDir = workDir, dataFile = dataFile, dataRunMax = dataRunMax, angleUS = angleUS)

    # Load MC
    TotalMCStatistics, nBKGs, channels = X17pythonTask_2023.readMC(channels, CUTfile = workDir + 'MC2023totOLDmerge.root:ntuple', workDir = workDir, MCFile = MCFile, ECODETYPE = ECODETYPE, X17masses = X17masses, dX17mass = dX17mass, alphares = alphares, alphafield = alphafield, esumCutLow = esumCutLow, esumCutHigh = esumCutHigh, angleCutLow = angleCutLow, angleCutHigh = angleCutHigh, BKGnames = BKGnames, alphaNames = alphaNames, scalingFactor = scalingFactor, simbeamEnergy = simbeamEnergy, angleUS = angleUS)

    alphavalues = [np.linspace(-5*alphares, 5*alphares, 11), np.linspace(-5*alphafield, 5*alphafield, 11)]
    alphaRefs = [0, 0]
    Hists = X17pythonTask_2023.histHandler(channels, 'dataHist', ['X17_17.6', 'X17_18.1'], BKGnames, 'Esum', 'Angle', alphaNames, alphavalues, alphaRefs, TotalMCStatistics=np.array(TotalMCStatistics), masses=X17masses, massRef=massRef)


    ################################################################################################################################################
    # Fit sidebands
    startingPars = np.array([100, 0.5, 16.9, 4e5, 1e4, 1e4, p176, p179, p181, 1, 1e4, 1e4, 0, 0, 0])
    FixedParameters = np.array([False, False, False, False, False, False, False, False, False, False, False, False, False, True, False])

    logL, betas, MAXLikelihood = X17pythonTask_2023.bestFit(startingPars, Hists, FitToy = False, doNullHypothesis = False, FixedParameters = FixedParameters)

    print(logL.values, logL.fval)
    
    #X17pythonTask_2023.plotComparison(Hists, logL.values, betas, channels, compareWithBetas=False, logL = logL, Toy = False)

    print(logL.values, logL.fcn(logL.values))

    BestBetas = np.copy(betas)
    BestPars = np.copy(logL.values)
    newP176 = logL.values[6]
    newP179 = logL.values[7]
    newP181 = logL.values[8]
    newAlphaField = logL.values[14]
    #_p176 = logL.values[6]
    #_p179 = logL.values[7]
    #_p181 = logL.values[8]
    #_alphaField = logL.values[14]

    print('Time elapsed to prepare the parameters: ', time.time() - startTime)
    
    ################################################################################################################################################
    # FC test
    
    # Signal grid point
    print('Start FC generation after ', time.time() - startTime, ' seconds')
    if generateDataSets:
        outputFileName = 'DataFC2023_N%d_p%.3f_m%.2f_NGrid%d_pGrid%.3f_MGrid%.2f_SEED%d.txt' % (DataSignalYield, DataSignalFraction, DataSignalMass, SignalYield, SignalFraction, SignalMass, DataSEED)
        with open(workDir + outputFileName, 'w') as f:
            f.write('#DataSignalYield DataSignalFraction DataSignalMass SignalYield SignalFraction SignalMass FitYield FitFraction FitMass MAXLikelihood locLikelihood datalRatio accurate valid DataSEED\n')
        for i in range(nToys):
            if i %10 == 0:
                print('Generating data toy number ', i + DataSEED, ' after ', time.time() - startTime, ' seconds')
            BestPars[0] = DataSignalYield
            BestPars[1] = DataSignalFraction
            BestPars[2] = DataSignalMass
            
            # New HistsTest
            HistsTest = X17pythonTask_2023.histHandler(channels, 'dataHist', ['X17_17.6', 'X17_18.1'], BKGnames, 'Esum', 'Angle', alphaNames, alphavalues, alphaRefs, TotalMCStatistics=np.array(TotalMCStatistics), masses=X17masses, massRef=massRef)
            
            np.random.seed(DataSEED + i)
            yields = np.concatenate([X17pythonTask_2023.getSignalYields(BestPars[0], BestPars[1]), X17pythonTask_2023.getYields(BestPars[3], BestPars[4], BestPars[5], BestPars[6], BestPars[7], BestPars[8], BestPars[9]), [BestPars[10], BestPars[11], BestPars[12]]])
            HistsTest.generateToy(yields, betas = BestBetas, fluctuateTemplates = True, morph = BestPars[-2:], mass=BestPars[2])
            
            # Need to sample center of constraints
            
            # Sample the nuisances
            if isinstance(FixedParameters, list) or isinstance(FixedParameters, np.ndarray):
                if FixedParameters[6] == False:
                    _p176 = np.random.normal(newP176, dP176)
                    while _p176 < 0 or _p176 > 1:
                        _p176 = np.random.normal(newP176, dP176)
                if FixedParameters[7] == False:
                    _p179 = np.random.normal(newP179, dP179)
                    while _p179 < 0 or _p179 > 1:
                        _p179 = np.random.normal(newP179, dP179)
                if FixedParameters[8] == False:
                    _p181 = np.random.normal(newP181, dP181)
                    while _p181 < 0 or _p181 > 1:
                        _p181 = np.random.normal(newP181, dP181)
                if FixedParameters[14] == False:
                    _alphaField = np.random.normal(newAlphaField, dAlphaField)
                    while _alphaField < Hists.alphavalues[1][0] or _alphaField > Hists.alphavalues[1][-1]:
                        _alphaField = np.random.normal(newAlphaField, dAlphaField)
            elif FixedParameters == False:
                _p176 = np.random.normal(newP176, dP176)
                while _p176 < 0 or _p176 > 1:
                    _p176 = np.random.normal(newP176, dP176)
                _p179 = np.random.normal(newP179, dP179)
                while _p179 < 0 or _p179 > 1:
                    _p179 = np.random.normal(newP179, dP179)
                _p181 = np.random.normal(newP181, dP181)
                while _p181 < 0 or _p181 > 1:
                    _p181 = np.random.normal(newP181, dP181)
                _alphaField = np.random.normal(newAlphaField, dAlphaField)
                while _alphaField < Hists.alphavalues[1][0] or _alphaField > Hists.alphavalues[1][-1]:
                    _alphaField = np.random.normal(newAlphaField, dAlphaField)
        
            
            HistsTest.DataArray = np.copy(HistsTest.DataArrayToy)
            HistsTest.SignalArray = np.copy(HistsTest.SignalArrayToy)
            HistsTest.SignalArrayNuisance = np.copy(HistsTest.SignalArrayNuisanceToy)
            HistsTest.SignalArrayNuisance5Sigma = np.copy(HistsTest.SignalArrayNuisance5SigmaToy)
            HistsTest.SignalArrayNuisance5SigmaArray = np.copy(HistsTest.SignalArrayNuisance5SigmaArrayToy)
            HistsTest.BKGarray = np.copy(HistsTest.BKGarrayToy)
            HistsTest.BKGarrayNuisance5Sigma = np.copy(HistsTest.BKGarrayNuisance5SigmaToy)
            HistsTest.BKGarrayNuisance5SigmaArray = np.copy(HistsTest.BKGarrayNuisance5SigmaArrayToy)
            
            SignalYield, SignalFraction, SignalMass, Toy, MAXLikelihood, locLikelihood, datalRatio, accurate, valid, SEED, FixedParameters, fitN, fitF, fitM = X17pythonTask_2023.FCgenerator(SignalYield, SignalFraction, SignalMass, logL, HistsTest, BestPars, SEED = SEED, nToys = 0, betas = BestBetas, fluctuateTemplates=True, FixedParameters = FixedParameters, PARS = [], Likelihood = [], Accurate = [], Valid = [], Toy = [], workDir = workDir, doingDataToy = True, oldP176 = _p176, oldP179 = _p179, oldP181 = _p181, oldAlphaField = _alphaField)
            with open(workDir + outputFileName, 'a') as f:
                f.write(f'{DataSignalYield} {DataSignalFraction} {DataSignalMass} {SignalYield} {SignalFraction} {SignalMass} {fitN} {fitF} {fitM} {MAXLikelihood} {locLikelihood} {datalRatio} {accurate} {valid} {DataSEED + i}\n')
    else:
        logL, betas, MAXLikelihood = X17pythonTask_2023.bestFit(startingPars, Hists, FitToy = False, doNullHypothesis = False, FixedParameters = FixedParameters)
        ns = np.linspace(-100, 100, 201)
        ys = []
        for i in range(len(ns)):
            ys.append(logL.fcn(np.concatenate([[ns[i]],logL.values[1:]])))

        #fig = plt.figure(figsize = (14, 14), dpi=100)
        #plt.plot(ns, ys)
        #plt.savefig('Check.png')


        #X17pythonTask_2023.plotComparison(Hists, logL.values, betas, channels, compareWithBetas=False, logL = logL, Toy = False, subfix='Test')
    
        print(BestPars)

        PARS, Likelihood, Accurate, Valid, Toy, FixedParameters = X17pythonTask_2023.FCgenerator(SignalYield, SignalFraction, SignalMass, logL, Hists, BestPars, SEED = SEED, nToys = nToys, betas = BestBetas, fluctuateTemplates=True, FixedParameters = FixedParameters, PARS = [], Likelihood = [], Accurate = [], Valid = [], Toy = [], workDir = workDir, oldP176 = p176, oldP179 = p179, oldP181 = p181, oldAlphaField = 0)
    print('Time elapsed to generate FC: ', time.time() - startTime)
