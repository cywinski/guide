# GUIDE: Guidance-based Incremental Learning with Diffusion Models
## Setup
###  Clone repo
```bash
git clone https://github.com/cywinski/continual-joint-diffusion.git
cd continual-joint-diffusion
```

### Prepare Conda environment
```bash
conda create -n diffusion_env python=3.8
conda activate diffusion_env
pip install .
```

### Login to wandb
```bash
wandb login
```

## Reproduction
Presented commands below are for single GPU setup. To run the training in distributed manner, run the same command with `mpiexec`:
```bash
mpiexec -n $NUM_GPUS python scripts.image_train ...
```
When training in a distributed manner, you must manually divide the `--batch_size` argument by the number of ranks. In lieu of distributed training, you may use `--microbatch 16` (or `--microbatch 1` in extreme memory-limited cases) to reduce memory usage.

### Diffusion models training

CIFAR-10 2 tasks:

```bash
python -m scripts.image_train --experiment_name=cifar10_ci2_class_cond_diffusion_100k --batch_size=256 --num_steps=100000 --dataset=CIFAR10 --num_tasks=2 --save_interval=100000 --gr_n_generated_examples_per_task=25000 --scale_classes_loss=False --grc_equal_n_examples_per_class=True --first_task_num_steps=100000 --seed=0 --timestep_respacing=1000 --use_ddim=False --plot_interval=50000 --log_interval=5000 --diffusion_steps_validation=1000 --cl_method=generative_replay --train_noised_classifier=False --train_with_disjoint_classifier=False --embedding_kind=concat_time_1hot --skip_validation=True --class_cond=True --num_res_blocks=3 --train_aug=True --classifier_augmentation=False
```

CIFAR-10 5 tasks:
```bash
python -m scripts.image_train --experiment_name=cifar10_ci5_class_cond_diffusion_50k --batch_size=256 --num_steps=50000 --dataset=CIFAR10 --num_tasks=5 --save_interval=50000 --gr_n_generated_examples_per_task=10000 --scale_classes_loss=False --grc_equal_n_examples_per_class=True --first_task_num_steps=100000 --seed=0 --timestep_respacing=1000 --use_ddim=False --plot_interval=50000 --log_interval=5000 --diffusion_steps_validation=1000 --cl_method=generative_replay --train_noised_classifier=False --train_with_disjoint_classifier=False --embedding_kind=concat_time_1hot --skip_validation=True --class_cond=True --num_res_blocks=3 --train_aug=True --classifier_augmentation=False
```

CIFAR-100 5 tasks:
```bash
python -m scripts.image_train --experiment_name=cifar100_ci5_class_cond_diffusion_50k --batch_size=256 --num_steps=50000 --dataset=CIFAR100 --num_tasks=5 --save_interval=50000 --gr_n_generated_examples_per_task=10000 --scale_classes_loss=False  --grc_equal_n_examples_per_class=True --first_task_num_steps=100000 --seed=0 --timestep_respacing=1000 --use_ddim=False --plot_interval=50000 --log_interval=5000 --diffusion_steps_validation=1000 --cl_method=generative_replay --train_noised_classifier=False --train_with_disjoint_classifier=False --embedding_kind=concat_time_1hot --skip_validation=True --class_cond=True --num_res_blocks=3 --train_aug=True --classifier_augmentation=False
```

CIFAR-100 10 tasks:
```bash
python -m scripts.image_train --experiment_name=cifar100_ci10_class_cond_diffusion_100k --batch_size=256 --num_steps=100000 --dataset=CIFAR100 --num_tasks=10 --save_interval=100000 --gr_n_generated_examples_per_task=5000 --scale_classes_loss=False  --grc_equal_n_examples_per_class=True --first_task_num_steps=100000 --seed=0 --timestep_respacing=1000 --use_ddim=False --plot_interval=50000 --log_interval=5000 --diffusion_steps_validation=1000 --cl_method=generative_replay --train_noised_classifier=False --train_with_disjoint_classifier=False --embedding_kind=concat_time_1hot --skip_validation=True --class_cond=True --num_res_blocks=3 --train_aug=True --classifier_augmentation=False
```

### Classifier trainings
To run classifier trainings you need to first train the diffusion models and store them in `--diffusion_pretrained_dir`.

CIFAR-10 2 tasks
```bash
python -m scripts.image_train --experiment_name=c10_ci2_ddim50_2ksteps --batch_size=256 --dataset=CIFAR10 --num_tasks=2 --scale_classes_loss=False --grc_equal_n_examples_per_class=True --seed=0 --timestep_respacing=ddim50 --use_ddim=True --classifier_scale_min_old=0.0 --classifier_scale_max_old=0.0 --classifier_scale_min_new=0.2 --classifier_scale_max_new=0.2 --norm_grads=False --trim_logits=True --cl_method=generative_replay_disjoint_classifier_guidance --train_with_disjoint_classifier=True --train_noised_classifier=False --use_old_grad=False --negate_old_grad=False --use_new_grad=True --guid_to_new_classes=True --embedding_kind=concat_time_1hot --class_cond=False --classifier_lr=0.01 --classifier_init_lr=0.1 --disjoint_classifier_init_num_steps=5000 --classifier_augmentation=True --disjoint_classifier_num_steps=2000 --log_interval=200 --plot_interval=10 --diffusion_pretrained_dir=<path/to/diffusion> --skip_validation=False --classifier_type=resnet
```

CIFAR-10 5 tasks
```bash
python -m scripts.image_train --experiment_name=c10_ci5_ddim50_2ksteps --batch_size=256 --dataset=CIFAR10 --num_tasks=5 --scale_classes_loss=False --grc_equal_n_examples_per_class=True --seed=0 --timestep_respacing=ddim50 --use_ddim=True --classifier_scale_min_old=0.0 --classifier_scale_max_old=0.0 --classifier_scale_min_new=0.2 --classifier_scale_max_new=0.2 --norm_grads=False --trim_logits=True --cl_method=generative_replay_disjoint_classifier_guidance --train_with_disjoint_classifier=True --train_noised_classifier=False --use_old_grad=False --negate_old_grad=True --use_new_grad=True --guid_to_new_classes=True --embedding_kind=concat_time_1hot --class_cond=False --classifier_lr=0.01 --classifier_init_lr=0.1 --disjoint_classifier_init_num_steps=5000 --classifier_augmentation=True --disjoint_classifier_num_steps=2000 --log_interval=200 --plot_interval=10 --diffusion_pretrained_dir=<path/to/diffusion> --skip_validation=False --classifier_type=resnet
```

CIFAR-100 5 tasks
```bash
python -m scripts.image_train --experiment_name=c100_ci5_ddim100_2ksteps_geninterval10 --batch_size=256 --dataset=CIFAR100 --num_tasks=5 --scale_classes_loss=False --grc_equal_n_examples_per_class=True --seed=0 --timestep_respacing=ddim100 --use_ddim=True --classifier_scale_min_old=0.0 --classifier_scale_max_old=0.0 --classifier_scale_min_new=0.2 --classifier_scale_max_new=0.2 --norm_grads=False --trim_logits=True --cl_method=generative_replay_disjoint_classifier_guidance --train_with_disjoint_classifier=True --train_noised_classifier=False --use_old_grad=False --negate_old_grad=True --use_new_grad=True --guid_to_new_classes=True --embedding_kind=concat_time_1hot --class_cond=False --classifier_lr=0.05 --classifier_init_lr=0.1 --classifier_weight_decay=5e-4 --disjoint_classifier_init_num_steps=10000 --classifier_augmentation=True --disjoint_classifier_num_steps=2000 --log_interval=500 --plot_interval=50 --diffusion_pretrained_dir=<path/to/diffusion> --skip_validation=False --classifier_type=resnet --guid_generation_interval=10
```

CIFAR-100 10 tasks
```bash
python -m scripts.image_train --experiment_name=c100_ci10_ddim100_2ksteps_geninterval10 --batch_size=256 --dataset=CIFAR100 --num_tasks=10 --scale_classes_loss=False --grc_equal_n_examples_per_class=True --seed=0 --timestep_respacing=ddim100 --use_ddim=True --classifier_scale_min_old=0.0 --classifier_scale_max_old=0.0 --classifier_scale_min_new=0.2 --classifier_scale_max_new=0.2 --norm_grads=False --trim_logits=True --cl_method=generative_replay_disjoint_classifier_guidance --train_with_disjoint_classifier=True --train_noised_classifier=False --use_old_grad=False --negate_old_grad=True --use_new_grad=True --guid_to_new_classes=True --embedding_kind=concat_time_1hot --class_cond=False --classifier_lr=0.05 --classifier_init_lr=0.1 --classifier_weight_decay=5e-4 --disjoint_classifier_init_num_steps=10000 --classifier_augmentation=True --disjoint_classifier_num_steps=2000 --log_interval=500 --plot_interval=50 --diffusion_pretrained_dir=<path/to/diffusion> --skip_validation=False --classifier_type=resnet --guid_generation_interval=10
```

## BibTeX
If you find this work useful, please consider citing it:
```bibtex

```

## Acknowledgments
This codebase borrows from [OpenAI's guided diffusion repo](https://github.com/openai/guided-diffusion) and [Continual-Learning-Benchmark repo](https://github.com/GT-RIPL/Continual-Learning-Benchmark).
