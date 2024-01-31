import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Settings for Improved Denoising Diffusion Models')
    parser.add_argument('--dataset', type=str,required=True, default='cifar10', choices=['mnist','triangles','cifar10'])
    parser.add_argument('--dataroot', type=str,required=True, help=' path to dataset directory')
    parser.add_argument('--batch_size', type=int, required=False, default=32)
    parser.add_argument('--image_size',type=int, default=32, required=False, help='define image size for triangles', choices = [32,64])
    parser.add_argument('--num_channels', type=int, default = 64, required = False, help='init channels for the Unet')
    parser.add_argument('--num_res_blocks' , type=int , default = 2, help='Number of residual blocks to use')
    parser.add_argument('--num_heads' , required=False, type=int , default = 4, help='The number of attention heads in each attention layer.')
    parser.add_argument('--attention_resolutions' , required=False,type=str, default = "16,8")
    parser.add_argument('--dropout', type=float, required=False, default = 0.0)
    parser.add_argument('--diffusion_steps', type=int, required=False, default=4000, help='Number of noisy diffusion steps')
    parser.add_argument('--learning_rate', type= float, required=False, default=1e-4)
    parser.add_argument('--num_heads_upsample', type = int, required = False, default = -1)
    parser.add_argument('--timestep_respacing', type = str, required = False, default="1000", help = "String containing comma-separated numbers, indicating the step count per section.")
    parser.add_argument('--training_steps', type=int, required=False, default=10000, help = 'Number of training iterations')
    return parser.parse_args()


def model_args(args):
    init_args = dict(image_size = 0, num_channels = 0, num_res_blocks = 0, num_heads = 0, num_heads_upsample = 0, attention_resolutions = '', dropout=0.0, diffusion_steps=0, timestep_respacing = '', dataset='')
    model_diffusion_args = {key: value for key,value in args.__dict__.items() if key in list(init_args.keys())}
    return model_diffusion_args

def training_args(args):
    init_args= dict(batch_size=0, learning_rate=1e-4, training_steps = 0)
    train_args = {key: value for key,value in args.__dict__.items() if key in list(init_args.keys())}
    return train_args

def load_data_args(args):
    init_args= dict(dataset='', dataroot='', batch_size = 0, image_size = 0)
    train_args = {key: value for key,value in args.__dict__.items() if key in list(init_args.keys())}
    return train_args

def split_args(args):
    model_diffusion_args = model_args(args)
    train_args = training_args(args)
    dataset_args = load_data_args(args)
    return model_diffusion_args, train_args, dataset_args
