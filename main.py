from utils.config import parse_args, split_args
import logging
from improved_diffusion.resample import create_named_schedule_sampler
from datasets.data_loader import load_data
from improved_diffusion.train_util import TrainLoop
from improved_diffusion.script_util import (
    create_model_and_diffusion)
import torch

def main(args):
    model_diffusion_args, training_model_args, dataset_args = split_args(args)
    train_loader, test_loader = load_data(**dataset_args)
    model, diffusion = create_model_and_diffusion(**model_diffusion_args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info (f'Running model on CUDA: GPU - {torch.cuda.get_device_name()}' if device.type =='cuda' else f' Running model on CPU: No GPU with CUDA available')
    model.to(device)
    schedule_sampler = create_named_schedule_sampler(diffusion)
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=train_loader,
        schedule_sampler=schedule_sampler,
        device=device,
        **training_model_args
    ).run_loop()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format = "%(asctime)s [%(levelname)-s] %(message)s")
    args = parse_args()
    main(args)
