from utils.config import parse_args
import logging
from datasets.data_loader import load_data

def main(args):
    logging.info('Starting .... ')
    train_loader, test_loader = load_data(args)
    print(1)
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format = "%(asctime)s [%(levelname)-s] %(message)s")
    args = parse_args()
    main(args)