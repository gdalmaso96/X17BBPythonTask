jobs = open('joblist1.txt', 'w')

# nRuns, offset
nTrials = 1000

offset = []

nSamples = 200
workDir = '/data/project/general/muonGroup/simulations/giovanni/X17BBPythonTask/results/'

command = []
## Normal run: 1e5 events per ecode 
## 000xxxx
#for i in range(10):
#    command.append(f'python3 -u /data/project/general/muonGroup/simulations/giovanni/X17BBPythonTask/testEstimators.py --workDir {workDir} --seed {i*nSamples} --nSamples {nSamples} --reset True\n')
#    offset.append(i)

# Current statistics run: 1e5 IPCs, 1e4 EPCs, 1e5 X17
# 001xxxx
for i in range(10):
    command.append(f'python3 -u /data/project/general/muonGroup/simulations/giovanni/X17BBPythonTask/testEstimators.py --workDir {workDir} --seed {i*nSamples} --nSamples {nSamples} --reset True --referenceFile X17referenceRealistic.root --prefix CurrentStatistics \n')
    offset.append(10000 + i)


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

