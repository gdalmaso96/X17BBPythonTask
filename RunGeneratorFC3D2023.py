import numpy as np
jobs = open('joblist.txt', 'w')


###############################################
# FC grid parameters
script = '/data/project/general/muonGroup/simulations/giovanni/X17BBPythonTask/generateFC3D2023.py'

################ Final grid ##################
yields = np.concatenate((np.arange(0, 351, 25), np.arange(400, 651, 50)))
masses = np.arange(16.5, 17.1, 0.05)
#masses = np.arange(16.5, 17.3, 0.05)
fractions = np.arange(0, 1.01, 0.25)
print(len(yields)*len(masses)*len(fractions))
###############################################

python = 'singularity run --bind /data/project/general/muonGroup/simulations/giovanni/:/data/project/general/muonGroup/simulations/giovanni /data/project/general/muonGroup/simulations/giovanni/Dockers/python3/python3.sif '

toysPerPoint = 1e3
toysPerJob = 20

command = []

nJobs = int(toysPerPoint/toysPerJob)
for i in range(nJobs):
    for m in range(len(masses)):
        for y in yields[:1]:
            for f in range(len(fractions)):
                command.append(f'python3 -u {script} --SignalYield {y} --SignalFraction {fractions[f]} --SignalMass {masses[m]} --SEED {int(i*toysPerJob + m*f*toysPerPoint)} --nToys {toysPerJob}')
        for y in yields[1:]:
            for f in fractions:
                command.append(f'python3 -u {script} --SignalYield {y} --SignalFraction {f} --SignalMass {masses[m]} --SEED {int(i*toysPerJob)} --nToys {toysPerJob}')

###############################################
# Data grid parameters
# Null signal, fraction 0
nDatas = 1e2
nDatasPerJob = 20
nJobs = int(nDatas/nDatasPerJob)
for i in range(nJobs):
    for m in masses:
        for y in yields:
            for f in fractions:
                command.append(f'python3 -u {script} --SignalYield {y} --SignalFraction {f} --SignalMass {m} --SEED {int(i*nDatasPerJob)} --nToys {int(nDatasPerJob)} --generateDataSets True --DataSEED {int(i*nDatasPerJob)} --DataSignalMass 16.9 --DataSignalFraction 0 --DataSignalYield 0')

# ATOMKI signal, fraction 1, signal = 72
nDatas = 1e2
nDatasPerJob = 20
nJobs = int(nDatas/nDatasPerJob)
for i in range(nJobs):
    for m in masses:
        for y in yields:
            for f in fractions:
                command.append(f'python3 -u {script} --SignalYield {y} --SignalFraction {f} --SignalMass {m} --SEED {int(i*nDatasPerJob)} --nToys {int(nDatasPerJob)} --generateDataSets True --DataSEED {int(i*nDatasPerJob)} --DataSignalMass 16.97 --DataSignalFraction 1 --DataSignalYield 74')

# ZM signal, fraction 0.29, signal = 255
nDatas = 1e2
nDatasPerJob = 20
nJobs = int(nDatas/nDatasPerJob)
for i in range(nJobs):
    for m in masses:
        for y in yields:
            for f in fractions:
                command.append(f'python3 -u {script} --SignalYield {y} --SignalFraction {f} --SignalMass {m} --SEED {int(i*nDatasPerJob)} --nToys {int(nDatasPerJob)} --generateDataSets True --DataSEED {int(i*nDatasPerJob)} --DataSignalMass 16.97 --DataSignalFraction 0.29 --DataSignalYield 255')

###############################################
# Write slurm files
directory = '/data/project/general/muonGroup/simulations/giovanni/X17BBPythonTask/slurm/'
for i in range(len(command)):
    run = f'run_{i:07}.sl'

    with open(directory + run, 'w') as slurm_file:
        slurm_file.write('#!/bin/bash\n')
        slurm_file.write('#SBATCH --cluster=merlin6\n')
        slurm_file.write('#SBATCH --partition=hourly\n')
        slurm_file.write('#SBATCH -o /data/project/general/muonGroup/simulations/giovanni/X17BBPythonTask/OUT/' + run[:-3] + '.out\n')
        slurm_file.write('#SBATCH -e /data/project/general/muonGroup/simulations/giovanni/X17BBPythonTask/OUT/' + run[:-3] + '.err\n')
        slurm_file.write('ulimit -c 0\n')
        slurm_file.write('echo Running on: `hostname`\n')
        slurm_file.write('TIMESTART=`date`\n')
        slurm_file.write('echo Start Time: ${TIMESTART}\n')
        slurm_file.write('echo ###################################################################\n')
        slurm_file.write('echo #                     Running Environement                        #\n')
        slurm_file.write('echo ###################################################################\n')
        slurm_file.write('env|sort\n')
        slurm_file.write('echo ###################################################################\n')
        slurm_file.write('echo #                 End of Running Environement                     #\n')
        slurm_file.write('echo ###################################################################\n')
        #slurm_file.write('source /psi/home/dalmaso_g/.bashrc\n')
        slurm_file.write('source /data/project/general/muonGroup/simulations/giovanni/.bashrc\n')
        slurm_file.write('\n')
        slurm_file.write('module unload psi-python38/2020.11\n')
        slurm_file.write('module load psi-python311/2024.02\n')
        slurm_file.write('\n')
        slurm_file.write('export OMP_NUM_THREADS=1\n')
        slurm_file.write('export MKL_NUM_THREADS=1\n')
        slurm_file.write('export OPENBLAS_NUM_THREADS=1\n')
        slurm_file.write('\n')
        slurm_file.write(command[i] + '\n')
        slurm_file.write('\n')
        slurm_file.write('echo Exit status: $?\n')
        slurm_file.write('echo Start Time: ${TIMESTART}\n')
        slurm_file.write('echo Stop Time: `date`\n')
    jobs.write('sbatch ' + directory + run + '\n')

jobs.close()

