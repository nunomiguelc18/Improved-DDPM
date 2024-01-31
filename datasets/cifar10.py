import torch
from torchvision import datasets
import pathlib
import logging
import numpy as np
import torch.nn.functional as F
class CIFAR10:
    def __init__(self, data_dir:str):
        self.data_dir = pathlib.Path(data_dir)

    def _download_cifar10(self) -> None:
        train_dataset = datasets.CIFAR10(root=self.data_dir, train=True, download=True)
        test_dataset = datasets.CIFAR10(root=self.data_dir, train=False, download=True)

        torch.save({
            'train_data': train_dataset.data,
            'train_labels': train_dataset.targets,
            'test_data': test_dataset.data,
            'test_labels': test_dataset.targets,
        }, f"{self.data_dir}/cifar10_data.pth")


    def _load(self) -> tuple([torch.Tensor,torch.Tensor]):
        if not (pathlib.Path(self.data_dir) / "cifar10_data.pth)").exists():
            self._download_cifar10()
        # Load MNIST dataset from the specified directory
        data = torch.load(f"{self.data_dir}/cifar10_data.pth")
        logging.info("CIFA10 dataset Loaded !")

        x_train = torch.tensor(data['train_data']).moveaxis(-1,1)
        y_train = data['train_labels']
        x_test = torch.tensor(data['test_data']).moveaxis(-1,1)
        y_test = data['test_labels']

        return (x_train, y_train), (x_test, y_test)