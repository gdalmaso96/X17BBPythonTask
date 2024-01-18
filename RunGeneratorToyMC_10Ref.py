jobs = open('joblist.txt', 'w')

# nRuns, offset
nTrials = 1000

offset = []

nSamplesPerRun = 1000
minnX17 = 450
maxnX17 = 900
npX17 = 15
minMassX17 = 15
maxMassX17 = 18
nMassX17 = 15
workDir = '/data/project/general/muonGroup/simulations/giovanni/X17BBPythonTask/results/'

command = []
# Ideal statistics run: 1e5 IPCs, 1e5 EPCs, 1e5 X17
# 004xxxx
for L in range(100):
    for i in range(int(nTrials/nSamplesPerRun)):
        for j in range(npX17):
            for k in range(nMassX17):
                nSamples = nSamplesPerRun
                command.append(f'python3 -u /data/project/general/muonGroup/simulations/giovanni/X17BBPythonTask/testEstimators.py --parametrizeX17 True --ToySample True --workDir {workDir} --seed {i*nSamples} --numberToys {nSamples} --resetFC True --referenceFile X17referenceIdeal{L}.root --prefix bins20x14Ideal{L}StatisticsParametrized --massX17 {j*(maxMassX17 - minMassX17)/(nMassX17 - 1) + minMassX17} --numberX17 {minnX17 + (maxnX17 - minnX17)/(npX17 - 1)*k}\n')
                offset.append(400000 + L*int(nTrials/nSamplesPerRun)*npX17*nMassX17 + i*npX17*nMassX17 + j*nMassX17 + k)

# Current statistics run: 1e5 IPCs, 1e4 EPCs, 1e5 X17
# 005xxxx
for L in range(100):
    for i in range(int(nTrials/nSamplesPerRun)):
        for j in range(npX17):
            for k in range(nMassX17):
                nSamples = nSamplesPerRun
                command.append(f'python3 -u /data/project/general/muonGroup/simulations/giovanni/X17BBPythonTask/testEstimators.py --parametrizeX17 True --ToySample True --workDir {workDir} --seed {i*nSamples} --numberToys {nSamples} --resetFC True --referenceFile X17referenceCurrent{L}.root --prefix bins20x14Current{L}StatisticsParametrized --massX17 {j*(maxMassX17 - minMassX17)/(nMassX17 - 1) + minMassX17} --numberX17 {minnX17 + (maxnX17 - minnX17)/(npX17 - 1)*k}\n')
                offset.append(500000 + L*int(nTrials/nSamplesPerRun)*npX17*nMassX17 + i*npX17*nMassX17 + j*nMassX17 + k)


for i in range(len(command)):
    run = offset[i]
    slurm_file = open('/data/project/general/muonGroup/simulations/giovanni/X17BBPythonTask/slurm/test%07i.sl' %run,'w')
    jobs.write('sbatch /data/project/general/muonGroup/simulations/giovanni/X17BBPythonTask/slurm/test%07i.sl\n' %run)
    
    slurm_file.write('#!/bin/bash\n')
    slurm_file.write('#SBATCH --cluster=merlin6 \n')
    slurm_file.write('#SBATCH --partition=hourly \n')
    slurm_file.write('#SBATCH -o /data/project/general/muonGroup/simulations/giovanni/X17BBPythonTask/OUT/test%07i.out \n' %run)
    slurm_file.write('#SBATCH -e /data/project/general/muonGroup/simulations/giovanni/X17BBPythonTask/OUT/test%07i.err \n' %run)
    slurm_file.write('#SBATCH --time 0-00:55:00\n\n')
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


