import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Settings for Improved Denoising Diffusion Models')
    parser.add_argument('--dataset', type=str,required=True, default='mnist', choices=['mnist','triangles'])
    parser.add_argument('--dataroot', type=str,required=True, help=' path to dataset directory')
    parser.add_argument('--batch_size', type=int, required=False, default=32)
    parser.add_argument('--image_size',type=int, default=32, required=False, help='define image size for triangles')

    return parser.parse_args()


