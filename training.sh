#!/bin/bash

#SBATCH  --time=1-23:29:59
#SBATCH --mem=30G
#SBATCH --cpus-per-task=1
#SBATCH --job-name=matrix_testing_bwe_blind_1000
#SBATCH  --gres=gpu:a100:1
##SBATCH  --gres=gpu:1 --constraint=volta
#SBATCH --output=/scratch/work/%u/projects/ddpm/blind_bwe_diffusion/train_%j.out

#SBATCH --array=[93]

module load anaconda
source activate /scratch/work/molinee2/conda_envs/pytorch2
module load gcc/8.4.0
export TORCH_USE_RTLD_GLOBAL=YES
#export HYDRA_FULL_ERROR=1
#export CUDA_LAUNCH_BLOCKING=1
n=$SLURM_ARRAY_TASK_ID

#n=94

namerun=training_filter_score_model
name="${n}_$namerun"
iteration=`sed -n "${n} p" iteration_parameters.txt`
#
PATH_EXPERIMENT=experiments/$n
mkdir $PATH_EXPERIMENT

#python train_w_cqt.py path_experiment="$PATH_EXPERIMENT"  $iteration
python train.py model_dir="$PATH_EXPERIMENT" $iteration
