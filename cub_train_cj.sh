#!/bin/bash
#SBATCH -A plgdiffusion-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH -t 480
#SBATCH --gpus=1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --nodes 1
#SBATCH -o slurm_out/slurm-%j.log
#SBATCH --job-name=cj-cub

module load GCC/11.2.0
module load OpenMPI/4.1.2-CUDA-11.6.0

eval "$(conda shell.bash hook)"
conda activate guide

which conda
which python3

/net/tscratch/people/plgbcywinski/conda/envs/guide/bin/python3 -m scripts.image_train_cub200_cj --wandb_experiment_name=cub200_CONTINUAL-JOINT --wandb_project_name=diffusion_guidance_cl --wandb_entity=cl-diffusion --batch_size=256 --microbatch=16 --num_tasks=1 --seed=0 --timestep_respacing=ddim25 --use_ddim=True --classifier_scale_min_new=0.0 --classifier_scale_max_new=0.0 --cl_method=generative_replay_disjoint_classifier_guidance --train_with_disjoint_classifier=True --use_old_grad=False --use_new_grad=False --guid_to_new_classes=True --classifier_lr=0.0001 --disjoint_classifier_init_num_steps=460 --classifier_augmentation=False --log_interval=1 --attention_resolutions 32,16,8 --resblock_updown True --use_scale_shift_norm True --num_channels 256 --num_head_channels 64 --learn_sigma True --dropout 0.0 --num_res_blocks 2 --resume_checkpoint=/net/pr2/projects/plgrid/plggdiffusion/checkpoints/256x256_diffusion.pt --dataroot=/net/pr2/projects/plgrid/plggdiffusion/
