from .mnist import MNIST
from .triangles import Generate_Triangles
from .scaler import Normalize
import logging
from torch.utils.data import DataLoader
def load_data(args) -> tuple[(DataLoader, DataLoader)]:
    logging.info(f'Loading dataset...')
    config_scaler = Normalize(-1,1)

    if args.dataset == 'mnist':
        training_dataset, testing_dataset = MNIST(args.dataroot)._load()
        training_dataset, testing_dataset = config_scaler.transform(training_dataset[0]), config_scaler.transform(testing_dataset[0])
    
    elif args.dataset == 'triangles':
        config_triangles = Generate_Triangles(args.image_size,args.image_size)
        training_dataset, testing_dataset = config_triangles.generate_dataset(N_training=80000,N_testing=20000)

    
    return DataLoader(config_scaler.transform(training_dataset),batch_size=args.batch_size,shuffle=True), DataLoader(config_scaler.transform(testing_dataset),batch_size=args.batch_size,shuffle=True)

    
