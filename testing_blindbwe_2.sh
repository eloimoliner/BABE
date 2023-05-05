#!/bin/bash
#SBATCH  --time=03:59:59
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

#maestro

#n=3 #original CQTDiff (with fast implementation) (22kHz)

n=54 #cqtdiff+ maestro 8s (alt version)
#n=50 #cqtdiff+ attention maestro 8s (alt version)

#n=51 #musicnet
#n=64 #cocochorales str
#n=65 #cocochorales strings
#n=93 #cocochorales brass
#n=94 #cocochorales woodwinf
#n=88 #maestro 22k
#n=87 #maestro 44k

if [[ $n -eq 54 ]] 
then
    ckpt="/scratch/work/molinee2/projects/ddpm/blind_bwe_diffusion/experiments/54_maestro_22k/22k_8s-850000.pt"
    exp=maestro22k_8s
    network=paper_1912_unet_cqt_oct_noattention_adaln
    #tester=blind_bwe_sweep
    #tester=blind_bwe_denoise
    #tester=blind_bwe_formal_3000_opt
    tester=bwe_formal_3000_opt_2
    #tester=edm_DC_correction_longer
    dset=maestro_allyears
    CQT=True
    diff_params=edm
elif [[ $n -eq 87 ]] 
then
    ckpt="/scratch/work/molinee2/projects/ddpm/diffusion_autumn_2022/A-diffusion/experiments/87/44k_8s-390000.pt"
    exp=maestro44k_8s
    network=unet_cqt_oct_maestro_44k
    tester=blind_bwe_44k
    #tester=edm_DC_correction_longer
    dset=maestro_allyears
    CQT=True
    diff_params=edm_44k
elif [[ $n -eq 88 ]] 
then
    ckpt="/scratch/work/molinee2/projects/ddpm/diffusion_autumn_2022/A-diffusion/experiments/88/22k_8s-1060000.pt"
    exp=maestro22k_8s
    network=paper_1912_unet_cqt_oct_noattention_adaln
    tester=blind_bwe
    #tester=edm_DC_correction_longer
    dset=maestro_allyears
    CQT=True
    diff_params=edm
elif [[ $n -eq 64 ]] 
then
    ckpt="/scratch/work/molinee2/projects/ddpm/diffusion_autumn_2022/A-diffusion/experiments/64/22k_8s-190000.pt"
    exp=CocoChorales_16k_8s
    network=paper_1912_unet_cqt_oct_noattention_adaln
    tester=blind_bwe_cocochorales
    #tester=edm_DC_correction_longer
    dset=CocoChorales
    CQT=True
    diff_params=edm_chorales
elif [[ $n -eq 65 ]] 
then
    ckpt="/scratch/work/molinee2/projects/ddpm/blind_bwe_diffusion/experiments/65/22k_8s-190000.pt"
    exp=CocoChorales_16k_8s
    network=paper_1912_unet_cqt_oct_noattention_adaln
    tester=blind_bwe_denoise_strings
    #tester=edm_DC_correction_longer
    dset=CocoChorales_stems
    CQT=True
    diff_params=edm_chorales
elif [[ $n -eq 93 ]] 
then
    ckpt="/scratch/work/molinee2/projects/ddpm/blind_bwe_diffusion/experiments/93/22k_8s-370000.pt"
    exp=CocoChorales_16k_8s
    network=paper_1912_unet_cqt_oct_noattention_adaln
    tester=blind_bwe_denoise_brass
    #tester=edm_DC_correction_longer
    dset=CocoChorales_stems
    CQT=True
    diff_params=edm_chorales
elif [[ $n -eq 94 ]] 
then
    ckpt="/scratch/work/molinee2/projects/ddpm/blind_bwe_diffusion/experiments/94/22k_8s-390000.pt"
    exp=CocoChorales_16k_8s
    network=paper_1912_unet_cqt_oct_noattention_adaln
    tester=blind_bwe_denoise_woodwind
    #tester=edm_DC_correction_longer
    dset=CocoChorales_stems
    CQT=True
    diff_params=edm_chorales
elif [[ $n -eq 61 ]] 
then
    ckpt="/scratch/work/molinee2/projects/ddpm/diffusion_autumn_2022/A-diffusion/experiments/61/vctk_16k_udm_500000.pt"
    exp=diffwave-sr_16k
    network=diffwave-sr_16k
    tester=blind_bwe_vctk
    #tester=edm_DC_correction_longer
    dset=vctk_test
    CQT=False
    diff_params=edm_eps
elif [[ $n -eq 63 ]] 
then
    ckpt="/scratch/work/molinee2/projects/ddpm/diffusion_autumn_2022/A-diffusion/experiments/63/22k_8s-210000.pt"
    exp=vctk_16k
    network=diffwave-sr_16k
    tester=blind_bwe_vctk2
    #tester=edm_DC_correction_longer
    dset=vctk
    CQT=False
    diff_params=edm
elif [[ $n -eq 3 ]] 
then
    ckpt="/scratch/work/molinee2/projects/ddpm/diffusion_autumn_2022/A-diffusion/experiments/3/weights-489999.pt"
    exp=test_cqtdiff_22k
    network=unet_cqtdiff_original
    tester=bwe_formal_1000_matrix
    dset=maestro_allyears
    CQT=False
    diff_params=edm


elif [[ $n -eq 56 ]] 
then
    ckpt="/scratch/work/molinee2/projects/ddpm/diffusion_autumn_2022/A-diffusion/experiments/56/22k_8s-510000.pt"
    exp=maestro22k_131072
    network=ADP_raw_patching
    tester=inpainting_tester
    dset=maestro_allyears
    CQT=False
elif [[ $n -eq 50 ]] 
then
    ckpt="/scratch/work/molinee2/projects/ddpm/diffusion_autumn_2022/A-diffusion/experiments/50/22k_8s-750000.pt"
    exp=maestro22k_8s
    network=paper_1912_unet_cqt_oct_attention_adaLN_2
    dset=maestro_allyears
    tester=blind_bwe
    CQT=True

elif [[ $n -eq 51 ]] 
then
    ckpt="/scratch/work/molinee2/projects/ddpm/diffusion_autumn_2022/A-diffusion/experiments/51/44k_4s-560000.pt"
    exp=musicnet44k_4s
    network=paper_1912_unet_cqt_oct_attention_44k_2
    dset=musicnet
    #dset=inpainting_musicnet
    tester=blind_bwe
    CQT=True
    diff_params=edm
fi


PATH_EXPERIMENT=experiments/blind_bwe_tests/$n
#PATH_EXPERIMENT=experiments/cqtdiff_original
mkdir $PATH_EXPERIMENT


#python train_w_cqt.py path_experiment="$PATH_EXPERIMENT"  $iteration
python test.py model_dir="$PATH_EXPERIMENT" \
               dset=$dset \
               exp=$exp \
               network=$network \
               tester=$tester \
               tester.checkpoint=$ckpt \
               tester.filter_out_cqt_DC_Nyq=$CQT \
               diff_params=$diff_params 
                
