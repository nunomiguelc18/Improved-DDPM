import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Settings for Improved Denoising Diffusion Models')
    parser.add_argument('--dataset', type=str,required=True, default='mnist', choices=['mnist','fashion-mnist','cifar'],  help =' Choose which dataset to use')
    parser.add_argument('--dataroot', type=str,required=True, help='Path for dataset')
    parser.add_argument('--batch_size', type=int, required=False, default=32, help='size of mini batch')

    return parser.parse_args()


