from .mnist import MNIST
from .cifar10 import CIFAR10
from .triangles import Generate_Triangles
from .scaler import Normalize
import logging
from torch.utils.data import DataLoader
def load_data(dataset:str, dataroot: str, image_size: int, batch_size: int):
    logging.info(f'Loading dataset...')
    config_scaler = Normalize(-1,1)

    if dataset == 'mnist':
        training_dataset, testing_dataset = MNIST(dataroot)._load()
        training_dataset, testing_dataset = config_scaler.transform(training_dataset[0]), config_scaler.transform(testing_dataset[0])
    
    if dataset == 'cifar10':
        training_dataset, testing_dataset = CIFAR10(dataroot)._load()
        training_dataset, testing_dataset = config_scaler.transform(training_dataset[0]), config_scaler.transform(testing_dataset[0])
    
    elif dataset == 'triangles':
        config_triangles = Generate_Triangles(image_size,image_size)
        training_dataset, testing_dataset = config_triangles.generate_dataset(N_training=2000,N_testing=1000)

    
    return DataLoader(config_scaler.transform(training_dataset),batch_size=batch_size,shuffle=True), DataLoader(config_scaler.transform(testing_dataset),batch_size=batch_size,shuffle=True)

    
