#!/bin/sh
#SBATCH --job-name=multipro      # create a short name for your job
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=60
#SBATCH --time 500:0
#SBATCH --qos bbdefault
#SBATCH --constraint=icelake
set -e

module purge; module load bluebear
module load Miniconda3/4.9.2
source /rds/bear-apps/2021a/EL8-cas/software/Miniconda3/4.9.2/etc/profile.d/conda.sh
conda init 

conda activate bo
pip install topsis-jamesfallon
pip install Box2D
pip install pygame
# pip install gpy==1.10.0
# pip install topsis==0.2
# pip install numpy==1.20.3
# pip install matplotlib==3.5.0
# pip install gpy==1.10.0
# pip install pydoe2==1.3.0
# pip install statsmodels==0.13.1
# pip install pygmo


export p=$SLURM_CPUS_PER_TASK # Size of multiprocessing pool
export N=30                  # Number of inputs


# multiprocessing_slurm.py sets pool size from the first argument
srun python ../multiprocessing_slurm.py $p $N


# python ../run_experiment.py -p ACKLEY_10 -b 250 -r 1 -a PF


