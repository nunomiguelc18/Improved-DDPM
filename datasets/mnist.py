import torch
from torchvision import datasets
import pathlib
import logging

class MNIST:
    def __init__(self, data_dir:str):
        self.data_dir = pathlib.Path(data_dir)

    def _download_mnist(self) -> None:
        train_dataset = datasets.MNIST(root=self.data_dir, train=True, download=True)
        test_dataset = datasets.MNIST(root=self.data_dir, train=False, download=True)

        torch.save({
            'train_data': train_dataset.data,
            'train_labels': train_dataset.targets,
            'test_data': test_dataset.data,
            'test_labels': test_dataset.targets,
        }, f"{self.data_dir}/mnist_data.pth")


    def _load(self) -> tuple([torch.Tensor,torch.Tensor]):
        if not (pathlib.Path(self.data_dir) / "mnist_data.pth)").exists():
            self._download_mnist()
        # Load MNIST dataset from the specified directory
        data = torch.load(f"{self.data_dir}/mnist_data.pth")
        logging.info("MNIST dataset Loaded !")

        x_train = data['train_data']
        y_train = data['train_labels']
        x_test = data['test_data']
        y_test = data['test_labels']

        return (x_train, y_train), (x_test, y_test)
