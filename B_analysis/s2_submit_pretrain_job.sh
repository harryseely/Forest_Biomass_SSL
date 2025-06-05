#!/bin/bash
#SBATCH --job-name="Pretrain OCNN"
#SBATCH --output=./slurm_logs/%N-%j.out
#SBATCH --error=./slurm_logs/%N-%j.err
#SBATCH --time=4:00:0
#SBATCH --mem=32G
#SBATCH --tasks-per-node=4
#SBATCH --nodes 30
#SBATCH --gres=gpu:4

# Set up environment variables
export N_GPUS=4 #MUST MATCH N GPUS ABOVE
export N_NODES=$SLURM_NNODES
export DATETIME=`date +%Y_%m_%d_%H_%M_%S`
export TORCH_NCCL_ASYNC_HANDLING=1

#Start recording time for env setup
env_setup_t0=$(date +%s)

# Load python
echo "Loading python modules..."
module load python/3.11

#Ensure we are in root dir
cd ~/scratch/analysis
echo "Creating virtual environment..."

# Create the virtual environment on each node : 
srun --ntasks $SLURM_NNODES --tasks-per-node=1 bash << EOF
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r slurm_requirements.txt
EOF

#Report time and packages
env_setup_t1=$(date +%s)
echo "Installed packages:"
pip freeze
elapsed_time=$((env_setup_t1 - env_setup_t0))
echo "Time taken to set up python environment: $elapsed_time seconds"

# Activate env only on main node                                                               
source $SLURM_TMPDIR/env/bin/activate;

# srun exports the current env, which contains $VIRTUAL_ENV and $PATH variables
srun python3 -m B_analysis.s2_ocnn_pretrain;