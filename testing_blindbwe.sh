#!/bin/bash
#SBATCH  --time=00:59:59
#SBATCH --mem=30G
#SBATCH --cpus-per-task=1
#SBATCH --job-name=matrix_testing_bwe_blind_1000
##SBATCH  --gres=gpu:a100:1
#SBATCH  --gres=gpu:1 --constraint=volta
#SBATCH --output=/scratch/work/%u/projects/ddpm/blind_bwe_diffusion/blind_bwe_evaluation/test_%j.out


module load anaconda
#source activate /scratch/work/molinee2/conda_envs/cqtdiff
source activate /scratch/work/molinee2/conda_envs/pytorch2
module load gcc/8.4.0
export TORCH_USE_RTLD_GLOBAL=YES
#export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
n=$SLURM_ARRAY_TASK_ID


#n=54 #cqtdiff+ maestro 8s 
#n=65 #cocochorales strings
n=93 #cocochorales brass
#n=94 #cocochorales woodwinf

if [[ $n -eq 54 ]] 
then
    #ckpt="/scratch/work/molinee2/projects/ddpm/blind_bwe_diffusion/experiments/54_maestro_22k/22k_8s-850000.pt"
    ckpt="./experiments/maestro_piano/MAESTRO_22k_8s-850000.pt"
    exp=maestro22k_8s
    network=cqtdiff+
    #tester=blind_bwe_sweep
    #tester=bwe_formal_noguidance_3000_opt_2
    #tester=blind_bwe_formal_small_3000
    tester=blind_bwe_denoise       
    #tester=blind_bwe_mushra
    #tester=bwe_formal_1000_opt_robustness_1
    #tester=edm_DC_correction_longer
    dset=maestro_allyears
    CQT=True
    diff_params=edm
elif [[ $n -eq 65 ]] 
then
    ckpt="./experiments/COCOChorales_strings/COCOChorales_strings_16k_11s-190000.pt"
    exp=CocoChorales_16k_8s
    network=cqtdiff+
    tester=blind_bwe_denoise_strings
    dset=CocoChorales_stems
    CQT=True
    diff_params=edm_chorales
elif [[ $n -eq 93 ]] 
then
    ckpt="./experiments/COCOChorales_brass/COCOChorales_brass_16k_11s-480000.pt"
    exp=CocoChorales_16k_8s
    network=cqtdiff+
    tester=blind_bwe_denoise_brass
    dset=CocoChorales_stems
    CQT=True
    diff_params=edm_chorales
elif [[ $n -eq 94 ]] 
then
    ckpt="./experiments/COCOChorales_woodwind/COCOChorales_woodwind_16k_11s-390000.pt"
    exp=CocoChorales_16k_8s
    network=cqtdiff+
    tester=blind_bwe_denoise_woodwind
    dset=CocoChorales_stems
    CQT=True
    diff_params=edm_chorales
fi


PATH_EXPERIMENT=experiments/blind_bwe_tests/$n


mkdir $PATH_EXPERIMENT


python test.py model_dir="$PATH_EXPERIMENT" \
               dset=$dset \
               exp=$exp \
               network=$network \
               tester=$tester \
               tester.checkpoint=$ckpt \
               tester.filter_out_cqt_DC_Nyq=$CQT \
               diff_params=$diff_params 
                
