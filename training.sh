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
#n=$SLURM_ARRAY_TASK_ID

#n=54 #cqtdiff+ maestro 8s (alt version)
#n=65 #cocochorales strings
n=93 #cocochorales brass
#n=94 #cocochorales woodwinf

namerun=training
name="${n}_$namerun"

#
PATH_EXPERIMENT=experiments/$n
mkdir $PATH_EXPERIMENT

if [[ $n -eq 54 ]] 
then
    #ckpt="/scratch/work/molinee2/projects/ddpm/blind_bwe_diffusion/experiments/54_maestro_22k/22k_8s-850000.pt"
    ckpt="./experiments/maestro_piano/MAESTRO_22k_8s-850000.pt"
    exp=maestro22k_8s
    network=cqtdiff+
    tester=basic_tester       
    dset=maestro_allyears
    CQT=True
    diff_params=edm
    logging=basic_logging

elif [[ $n -eq 65 ]] 
then
    ckpt="./experiments/COCOChorales_strings/COCOChorales_strings_16k_11s-190000.pt"
    exp=CocoChorales_16k_8s
    network=cqtdiff+
    tester=basic_tester       
    dset=CocoChorales_stems
    CQT=True
    diff_params=edm_chorales
    logging=basic_logging
elif [[ $n -eq 93 ]] 
then
    ckpt="./experiments/COCOChorales_woodwind/COCOChorales_woodwind_22k_8s-480000.pt"
    exp=CocoChorales_16k_8s
    network=cqtdiff+
    tester=basic_tester       
    dset=CocoChorales_stems
    CQT=True
    diff_params=edm_chorales
    logging=basic_logging
elif [[ $n -eq 94 ]] 
then
    ckpt="./experiments/COCOChorales_brass/COCOChorales_brass_22k_8s-390000.pt"
    exp=CocoChorales_16k_8s
    network=cqtdiff+
    tester=basic_tester       
    dset=CocoChorales_stems
    CQT=True
    diff_params=edm_chorales
    logging=basic_logging
fi

#python train_w_cqt.py path_experiment="$PATH_EXPERIMENT"  $iteration


python train.py model_dir="$PATH_EXPERIMENT" dset=$dset exp=$exp network=$network diff_params=$diff_params tester=$tester logging=$logging exp.batch=4
