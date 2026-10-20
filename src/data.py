"""MNIST data with the FIXED train/validation split from decision D-008. UNTESTED starter code.

The test set is never loaded here: it is evaluated exactly once, at the end of Stage 7.
"""
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_data(batch_size, seed, root="data"):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda t: t.view(-1)),
    ])
    full = datasets.MNIST(root, train=True, download=True, transform=tfm)
    g = torch.Generator().manual_seed(0)              # FIXED split across all runs (D-008)
    train, val = random_split(full, [55000, 5000], generator=g)
    train_loader = DataLoader(train, batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed), drop_last=True)
    val_loader = DataLoader(val, 1000)
    return train_loader, val_loader
