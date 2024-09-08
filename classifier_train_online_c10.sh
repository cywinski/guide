#!/bin/bash
#SBATCH -A plgdynamic2-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH -t 24:00:00
#SBATCH --gpus=1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --nodes 1
#SBATCH -o slurm_out/slurm-%j.log
#SBATCH --job-name=c10online
#SBATCH --mail-type=BEGIN

module load GCC/11.2.0
module load OpenMPI/4.1.2-CUDA-11.6.0

eval "$(conda shell.bash hook)"
conda activate guide_env

which conda
which python3

/net/tscratch/people/plgbcywinski/conda/envs/guide/bin/python3 -m scripts.image_train_online --wandb_experiment_name=c10_ci5_ddim50_newTonew05_GUIDE_online_interval1_aug --wandb_project_name=diffusion_guidance_cl --wandb_entity=cl-diffusion --batch_size=10 --dataset=CIFAR10 --num_tasks=5 --seed=0 --timestep_respacing=ddim50 --use_ddim=True --classifier_scale_min_new=0.5 --classifier_scale_max_new=0.5 --cl_method=generative_replay_disjoint_classifier_guidance --train_with_disjoint_classifier=True --use_old_grad=False --use_new_grad=True --guid_to_new_classes=True --embedding_kind=concat_time_1hot --classifier_init_lr=0.1 --classifier_lr=0.1 --num_epochs=1 --classifier_augmentation=True --log_interval=100 --guid_generation_interval=1 --standard_norm_stats=False --diffusion_pretrained_dir=/net/tscratch/people/plgbcywinski/code/continual-joint-diffusion/diffusion/cifar10_ci5_class_cond_diffusion_50k
