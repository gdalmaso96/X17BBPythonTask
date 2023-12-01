import numpy as np
jobs = open('joblist.txt', 'w')

# nRuns, offset
nTrials = 100

offset = []

nSamplesPerRun = 2
minnX17 = 0
maxnX17 = 900
npX17 = 15
minMassX17 = 15
maxMassX17 = 18
nMassX17 = 15
workDir = '/data/project/general/muonGroup/simulations/giovanni/X17BBPythonTask/results/'

command = []
# Current statistics run: 1e5 IPCs, 1e4 EPCs, 1e5 X17
# 020xxxx
N = np.linspace(minnX17, maxnX17, 10)
for i in range(len(N)):
    for j in range(int(nTrials/nSamplesPerRun)):
        nSamples = nSamplesPerRun
        command.append(f'python3 -u /data/project/general/muonGroup/simulations/giovanni/X17BBPythonTask/testEstimators.py --seed 0 --referenceFile X17referenceRealistic.root --profileLikelihood True --profileLikelihood2D True --prefix bins20x14CurrentStatisticsParametrizedNull{N[i]} --parametrizeX17 True --dataFile X17MC2021_{N[i]} --mX17plMin {minMassX17} --numberPL {npX17} --numberX17 {N[i]} --numberToys {nSamplesPerRun} --workDir {workDir} --seed {j*nSamplesPerRun}\n')
        offset.append(200000 + i*int(nTrials/nSamplesPerRun) + j)

# Ideal statistics run: 1e5 IPCs, 1e5 EPCs, 1e5 X17
# 021xxxx
for i in range(len(N)):
    for j in range(int(nTrials/nSamplesPerRun)):
        nSamples = nSamplesPerRun
        command.append(f'python3 -u /data/project/general/muonGroup/simulations/giovanni/X17BBPythonTask/testEstimators.py --seed 0 --referenceFile X17reference.root --profileLikelihood True --profileLikelihood2D True --prefix bins20x14IdealStatisticsParametrizedNull{N[i]} --parametrizeX17 True --dataFile X17MC2021_{N[i]} --mX17plMin {minMassX17} --numberPL {npX17} --numberX17 {N[i]} --numberToys {nSamplesPerRun} --workDir {workDir} --seed {j*nSamplesPerRun + 299792458}\n')
        offset.append(210000 + i*int(nTrials/nSamplesPerRun) + j)


for i in range(len(command)):
    run = offset[i]
    slurm_file = open('/data/project/general/muonGroup/simulations/giovanni/X17BBPythonTask/slurm/test%07i.sl' %run,'w')
    jobs.write('sbatch /data/project/general/muonGroup/simulations/giovanni/X17BBPythonTask/slurm/test%07i.sl\n' %run)
    
    slurm_file.write('#!/bin/bash\n')
    slurm_file.write('#SBATCH --cluster=merlin6 \n')
    slurm_file.write('#SBATCH --partition=daily \n')
    slurm_file.write('#SBATCH -o /data/project/general/muonGroup/simulations/giovanni/X17BBPythonTask/OUT/test%07i.out \n' %run)
    slurm_file.write('#SBATCH -e /data/project/general/muonGroup/simulations/giovanni/X17BBPythonTask/OUT/test%07i.err \n' %run)
    slurm_file.write('ulimit -c 0\n')
    slurm_file.write('echo Running on: `hostname` \n')
    slurm_file.write('TIMESTART=`date`\n')
    slurm_file.write('echo Start Time: ${TIMESTART}\n')
    slurm_file.write('echo ###################################################################\n')
    slurm_file.write('echo #                     Running Environement                        #\n')
    slurm_file.write('echo ###################################################################\n')
    slurm_file.write('env|sort\n')
    slurm_file.write('echo ###################################################################\n')
    slurm_file.write('echo #                 End of Running Environement                     #\n')
    slurm_file.write('echo ###################################################################\n')
    slurm_file.write('source /data/project/general/muonGroup/simulations/giovanni/.bashrc\n')
    slurm_file.write(command[i])
    slurm_file.write('echo Exit status: $?\n')
    slurm_file.write('echo Start Time: ${TIMESTART}\n')
    slurm_file.write('echo Stop Time: `date`\n')
    slurm_file.close()


jobs.close()


