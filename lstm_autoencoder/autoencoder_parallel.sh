#!/bin/bash                      
#
#SBATCH -t 10:00:00          # walltime = 10 hours
#SBATCH -N 6                 # one node
#SBATCH --array=1-100
python core.py ${SLURM_ARRAY_TASK_ID}