"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import os
import numpy as np
import torch
import torch.distributed as dist
import logging
from improved_diffusion.script_util import create_model_and_diffusion
from utils.config import parse_args, split_args
import pathlib
import matplotlib.pyplot as plt

def main(args):
    model_diffusion_args, _, _, sample_args = split_args(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')

    logging.info (f'Generating samples on CUDA: GPU - {torch.cuda.get_device_name()}' if device.type =='cuda' else f' Running model on CPU: No GPU with CUDA available')

    model, diffusion = create_model_and_diffusion(**model_diffusion_args)
    model.load_state_dict(torch.load(pathlib.Path(sample_args['model_path']).resolve()))
    model.to(device)
    model.eval()

    logging.info('Generating Sanples ...')
    all_images = []
    while len(all_images) * sample_args['batch_size'] < sample_args['number_samples']:
        model_kwargs = {}
        sample_fn = diffusion.p_sample_loop
        color_code = 1 if args.dataset in ['mnist','triangles'] else 3
        sample = sample_fn(
            model,
            (sample_args['batch_size'], color_code, args.image_size, args.image_size),
            clip_denoised= True,
            model_kwargs=model_kwargs,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        print(sample.shape[0])
        if color_code == 1:
            for i in range (sample.shape[0]):
                plt.imshow(sample[i,:,:,0].cpu().numpy(), cmap='gray')
                plt.savefig(f'./generated_images/sample_{i}_c1.png')
                plt.close()
            break
        else:
            for i in range (sample.shape[0]):
                plt.imshow(sample[i,...].cpu().numpy())
                plt.savefig(f'./generated_images/sample_{i}_c3.png')
                plt.close()
            break


    # arr = np.concatenate(all_images, axis=0)
    # arr = arr[: args.num_samples]
    # if dist.get_rank() == 0:
    #     shape_str = "x".join([str(x) for x in arr.shape])
    #     out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
    #     logger.log(f"saving to {out_path}")
    #     if args.class_cond:
    #         np.savez(out_path, arr, label_arr)
    #     else:
    #         np.savez(out_path, arr)

    # dist.barrier()
    # logger.log("sampling complete")




if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format = "%(asctime)s [%(levelname)-s] %(message)s")
    args = parse_args()
    main(args)