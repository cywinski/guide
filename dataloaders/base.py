import os

import kornia as K
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .wrapper import CacheClassLabel


class FastDataset(Dataset):
    def __init__(self, data, labels):
        self.dataset = data
        self.labels = labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index], self.labels[index]


def CIFAR10(
    dataroot, skip_normalization=False, train_aug=False, classifier_augmentation=False
):
    train_transform_clf = None
    train_transform_diff = None
    # augmentation for diffusion training
    if train_aug:
        train_transform_diff = K.augmentation.ImageSequential(
            K.augmentation.RandomHorizontalFlip(),
        )

    # augmentation for classifier training
    if classifier_augmentation:
        train_transform_clf = K.augmentation.ImageSequential(
            K.augmentation.Denormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            K.augmentation.RandomCrop((32, 32), padding=4),
            K.augmentation.RandomRotation(30),
            K.augmentation.RandomHorizontalFlip(),
            K.augmentation.ColorJiggle(0.1, 0.1, 0.1, 0.1),
            K.augmentation.RandomErasing(scale=(0.1, 0.5)),
            K.augmentation.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        )

    target_transform = transforms.Lambda(lambda y: torch.eye(10)[y])

    train_dataset = torchvision.datasets.CIFAR10(
        root=dataroot,
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
        target_transform=target_transform,
    )
    train_dataset = CacheClassLabel(
        train_dataset,
        target_transform=target_transform,
    )

    val_dataset = torchvision.datasets.CIFAR10(
        root=dataroot,
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
        target_transform=target_transform,
    )
    val_dataset = CacheClassLabel(
        val_dataset,
        target_transform=target_transform,
    )
    print("Loading data")
    save_path = f"{dataroot}/fast_cifar10_train"
    if os.path.exists(save_path):
        fast_cifar_train = torch.load(save_path)
    else:
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
        data = next(iter(train_loader))
        fast_cifar_train = FastDataset(data[0], data[1])
        torch.save(fast_cifar_train, save_path)

    save_path = f"{dataroot}/fast_cifar10_val"
    if os.path.exists(save_path):
        fast_cifar_val = torch.load(save_path)
    else:
        val_loader = DataLoader(val_dataset, batch_size=len(val_dataset))
        data = next(iter(val_loader))
        fast_cifar_val = FastDataset(data[0], data[1])
        torch.save(fast_cifar_val, save_path)

    return (
        fast_cifar_train,
        fast_cifar_val,
        32,
        3,
        train_transform_clf,
        train_transform_diff,
        10,
    )


def CIFAR100(
    dataroot, skip_normalization=False, train_aug=False, classifier_augmentation=False
):
    train_transform_clf = None
    train_transform_diff = None
    # augmentation for diffusion training
    if train_aug:
        train_transform_diff = K.augmentation.ImageSequential(
            K.augmentation.RandomHorizontalFlip(),
        )

    # augmentation for classifier training
    if classifier_augmentation:
        train_transform_clf = K.augmentation.ImageSequential(
            K.augmentation.Denormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            K.augmentation.RandomCrop((32, 32), padding=4),
            K.augmentation.RandomRotation(30),
            K.augmentation.RandomHorizontalFlip(),
            K.augmentation.ColorJiggle(0.1, 0.1, 0.1, 0.1),
            K.augmentation.RandomErasing(scale=(0.1, 0.5)),
            K.augmentation.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        )

    target_transform = transforms.Lambda(lambda y: torch.eye(100)[y])

    train_dataset = torchvision.datasets.CIFAR100(
        root=dataroot,
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
        target_transform=target_transform,
    )
    train_dataset = CacheClassLabel(
        train_dataset,
        target_transform=target_transform,
    )

    val_dataset = torchvision.datasets.CIFAR100(
        root=dataroot,
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
        target_transform=target_transform,
    )
    val_dataset = CacheClassLabel(
        val_dataset,
        target_transform=target_transform,
    )

    print("Loading data")
    save_path = f"{dataroot}/fast_cifar100_train"
    if os.path.exists(save_path):
        fast_cifar_train = torch.load(save_path)
    else:
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
        data = next(iter(train_loader))
        fast_cifar_train = FastDataset(data[0], data[1])
        torch.save(fast_cifar_train, save_path)

    save_path = f"{dataroot}/fast_cifar100_val"
    if os.path.exists(save_path):
        fast_cifar_val = torch.load(save_path)
    else:
        val_loader = DataLoader(val_dataset, batch_size=len(val_dataset))
        data = next(iter(val_loader))
        fast_cifar_val = FastDataset(data[0], data[1])
        torch.save(fast_cifar_val, save_path)

    return (
        fast_cifar_train,
        fast_cifar_val,
        32,
        3,
        train_transform_clf,
        train_transform_diff,
        100,
    )


def ImageNet100(
    dataroot, skip_normalization=False, train_aug=False, classifier_augmentation=False
):
    train_transform_clf = None
    train_transform_diff = None
    # augmentation for diffusion training
    if train_aug:
        train_transform_diff = K.augmentation.ImageSequential(
            K.augmentation.RandomHorizontalFlip(),
        )

    print("Loading data")
    save_path = f"{dataroot}/fast_imagenet100_train"
    if os.path.exists(save_path):
        fast_imagenet_train = torch.load(save_path)
    else:
        target_transform = transforms.Lambda(lambda y: torch.eye(100)[y])

        train_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(dataroot, "imagenet100", "train"),
            transform=transforms.Compose(
                [
                    transforms.Resize((76, 76)),
                    transforms.CenterCrop((64, 64)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
            target_transform=target_transform,
        )
        train_dataset = CacheClassLabel(
            train_dataset,
            target_transform=target_transform,
        )

        val_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(dataroot, "imagenet100", "val"),
            transform=transforms.Compose(
                [
                    transforms.Resize((76, 76)),
                    transforms.CenterCrop((64, 64)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
            target_transform=target_transform,
        )

        val_dataset = CacheClassLabel(
            val_dataset,
            target_transform=target_transform,
        )

        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
        data = next(iter(train_loader))
        fast_imagenet_train = FastDataset(data[0], data[1])
        torch.save(fast_imagenet_train, save_path)

    save_path = f"{dataroot}/fast_imagenet100_val"
    if os.path.exists(save_path):
        fast_imagenet_val = torch.load(save_path)
    else:
        val_loader = DataLoader(val_dataset, batch_size=len(val_dataset))
        data = next(iter(val_loader))
        fast_imagenet_val = FastDataset(data[0], data[1])
        torch.save(fast_imagenet_val, save_path)

    return (
        fast_imagenet_train,
        fast_imagenet_val,
        64,
        3,
        train_transform_clf,
        train_transform_diff,
        100,
    )


def ImageNet100128(
    dataroot, skip_normalization=False, train_aug=False, classifier_augmentation=False
):
    train_transform_clf = None
    train_transform_diff = None
    # augmentation for diffusion training
    if train_aug:
        train_transform_diff = K.augmentation.ImageSequential(
            K.augmentation.RandomHorizontalFlip(),
        )

    print("Loading data")
    save_path = f"{dataroot}/fast_imagenet100128_train"
    if os.path.exists(save_path):
        fast_imagenet_train = torch.load(save_path)
    else:
        target_transform = transforms.Lambda(lambda y: torch.eye(100)[y])

        train_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(dataroot, "imagenet100", "train"),
            transform=transforms.Compose(
                [
                    transforms.Resize((152, 152)),
                    transforms.CenterCrop((128, 128)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
            target_transform=target_transform,
        )
        train_dataset = CacheClassLabel(
            train_dataset,
            target_transform=target_transform,
        )

        val_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(dataroot, "imagenet100", "val"),
            transform=transforms.Compose(
                [
                    transforms.Resize((152, 152)),
                    transforms.CenterCrop((128, 128)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
            target_transform=target_transform,
        )

        val_dataset = CacheClassLabel(
            val_dataset,
            target_transform=target_transform,
        )

        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
        data = next(iter(train_loader))
        fast_imagenet_train = FastDataset(data[0], data[1])
        torch.save(fast_imagenet_train, save_path)

    save_path = f"{dataroot}/fast_imagenet100128_val"
    if os.path.exists(save_path):
        fast_imagenet_val = torch.load(save_path)
    else:
        val_loader = DataLoader(val_dataset, batch_size=len(val_dataset))
        data = next(iter(val_loader))
        fast_imagenet_val = FastDataset(data[0], data[1])
        torch.save(fast_imagenet_val, save_path)

    return (
        fast_imagenet_train,
        fast_imagenet_val,
        128,
        3,
        train_transform_clf,
        train_transform_diff,
        100,
    )
