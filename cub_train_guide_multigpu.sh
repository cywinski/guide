#!/bin/bash
#SBATCH -A plgdynamic2-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH -t 2880
#SBATCH --gpus=4
#SBATCH --ntasks 4
#SBATCH --cpus-per-task 8
#SBATCH --mem 128G
#SBATCH --nodes 1
#SBATCH -o slurm_out/slurm-%j.log
#SBATCH --job-name=4guide-cub05

module load GCC/11.2.0
module load OpenMPI/4.1.2-CUDA-11.6.0

eval "$(conda shell.bash hook)"
conda activate guide

which conda
which python3

mpiexec -n 4 /net/tscratch/people/plgbcywinski/conda/envs/guide/bin/python3 -m scripts.image_train_cub200 --wandb_experiment_name=cub200_ddpm250_GUIDE05 --wandb_project_name=diffusion_guidance_cl --wandb_entity=cl-diffusion --batch_size=64 --microbatch=4 --num_tasks=1 --seed=0 --timestep_respacing=250 --use_ddim=False --classifier_scale_min_new=0.5 --classifier_scale_max_new=0.5 --cl_method=generative_replay_disjoint_classifier_guidance --train_with_disjoint_classifier=True --use_old_grad=False --use_new_grad=True --guid_to_new_classes=True --classifier_init_lr=0.1 --classifier_lr=0.0001 --disjoint_classifier_init_num_steps=460 --classifier_augmentation=False --log_interval=10 --attention_resolutions 32,16,8 --resblock_updown True --use_scale_shift_norm True --num_channels 256 --num_head_channels 64 --learn_sigma True --dropout 0.0 --num_res_blocks 2 --resume_checkpoint=/net/pr2/projects/plgrid/plggdiffusion/checkpoints/256x256_diffusion.pt --dataroot=/net/pr2/projects/plgrid/plggdiffusion/
