import os

import torchvision
from torchvision import transforms
from .wrapper import CacheClassLabel
import torch
from torch.utils.data import Dataset, DataLoader
import kornia as K


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


class HFTinyImageNetDataset(Dataset):
    def __init__(
        self,
        root,
        name,
        split,
        transform=None,
        target_transform=None,
    ):
        import datasets

        self.root = root
        self.dataset = datasets.load_dataset(name, split=split)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label = self.dataset[index]["image"], self.dataset[index]["label"]
        img = img.convert("RGB")

        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            label = self.target_transform(label)

        return img, label


def TinyImageNet(
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
            K.augmentation.RandomCrop((64, 64), padding=4),
            K.augmentation.RandomRotation(30),
            K.augmentation.RandomHorizontalFlip(),
            K.augmentation.ColorJiggle(0.1, 0.1, 0.1, 0.1),
            K.augmentation.RandomErasing(scale=(0.1, 0.5)),
            K.augmentation.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        )

    target_transform = transforms.Lambda(lambda y: torch.eye(200)[y])

    train_dataset = HFTinyImageNetDataset(
        root=dataroot,
        name="Maysee/tiny-imagenet",
        split="train",
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

    val_dataset = HFTinyImageNetDataset(
        root=dataroot,
        name="Maysee/tiny-imagenet",
        split="valid",
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
    save_path = f"{dataroot}/fast_tinyimagenet_train"
    if os.path.exists(save_path):
        fast_tinyimagenet_train = torch.load(save_path)
    else:
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
        data = next(iter(train_loader))
        fast_tinyimagenet_train = FastDataset(data[0], data[1])
        torch.save(fast_tinyimagenet_train, save_path)

    save_path = f"{dataroot}/fast_tinyimagenet_val"
    if os.path.exists(save_path):
        fast_tinyimagenet_val = torch.load(save_path)
    else:
        val_loader = DataLoader(val_dataset, batch_size=len(val_dataset))
        data = next(iter(val_loader))
        fast_tinyimagenet_val = FastDataset(data[0], data[1])
        torch.save(fast_tinyimagenet_val, save_path)

    return (
        fast_tinyimagenet_train,
        fast_tinyimagenet_val,
        64,
        3,
        train_transform_clf,
        train_transform_diff,
        200,
    )
