#!/bin/bash

#SBATCH -J "clustering_agent_latent_space_name"
#SBATCH --time 48:00:00
#SBATCH --mem-per-gpu 50000

#SBATCH --partition gpu_all
#SBATCH --gres gpu:1

#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=8
#SBATCH --output criann_logs/%J_%x.out
#SBATCH --error  criann_logs/%J_%x.err

PROJECTNAME=Maxalaar_Reinforcement_Learning
module purge
module load aidl/pytorch/2.2.0-cuda12.1
export PYTHONUSERBASE=~/packages/$PROJECTNAME
export PATH=$PATH:~/packages/$PROJECTNAME/

# rsync -av --exclude='./temporary' --exclude='./experiments' --exclude='criann_logs' . $LOCAL_WORK_DIR
#echo Working directory : $PWD
#echo Local Working directory : $LOCAL_WORK_DIR
#
#cp -R . $LOCAL_WORK_DIR
#cd $LOCAL_WORK_DIR || exit
#echo Working directory : $PWD

params=(
  --experimentation_configuration_file ./configurations/criann/tetris_ppo_dense.py
)

ls
PYTHONPATH=$(pwd) srun python3 scripts/reinforcement_learning_training.py ${params[@]}
