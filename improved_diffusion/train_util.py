import copy
import functools
import os
import numpy as np
import torch
import torch.distributed as dist
from torch.optim import AdamW
from .resample import UniformSampler
import logging
import matplotlib.pyplot as plt
class TrainLoop:
    def __init__(
        self,
        model,
        diffusion,
        data,
        schedule_sampler,
        device,
        batch_size,
        learning_rate,
        training_steps
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.lr = learning_rate
        self.schedule_sampler = schedule_sampler
        self.step = 1
        self.training_steps = training_steps
        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.sync_cuda = torch.cuda.is_available()
        self.opt = AdamW(self.master_params, lr=self.lr)
        self.batch_iter = self.batch_generator()
        self.device = device
        self.cache_losses = []
    def batch_generator(self):
        while True:
             yield from self.data

    def run_loop(self):
        while (self.step < self.training_steps
        ):
            self.batch = next(self.batch_iter).to(self.device)
            self.run_step()
            # if self.step % self.log_interval == 0:
            #     logger.dumpkvs()
            # if self.step % self.save_interval == 0:
            #     self.save()
            #     # Run for a finite amount of time in integration tests.
            #     if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
            #         return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        # if (self.step - 1) % self.save_interval != 0:
        #     self.save()
     
    def run_step(self):
        self.forward_backward()
        self.optimize_normal()

    def forward_backward(self):
        
        self.opt.zero_grad()
        t, weights = self.schedule_sampler.sample(self.batch.shape[0], self.device)
        compute_losses = functools.partial(
            self.diffusion.training_losses,
            self.model,
            self.batch,
            t
        )

        losses = compute_losses()
        

        loss = (losses["loss"] * weights).mean()
        self.cache_losses.append(loss.item())
        if self.step % 10 == 0:
            logging.info(f"Step - {self.step}, losses : vb = {losses['vb'].mean().item(): .3f} , mse = {losses['mse'].mean().item(): .3f}, TOTAL = {loss.item(): .3f}")
            plt.plot(np.arange(self.step)+1,np.log(self.cache_losses))
            plt.ylabel('Training Loss - log scale')
            plt.savefig("./")
            plt.close()

        # log_loss_dict(
        #     self.diffusion, t, {k: v * weights for k, v in losses.items()}
        # )
        
        loss.backward()

    def optimize_normal(self):
        self._log_grad_norm()
        self.opt.step()
        
    def _log_grad_norm(self):
        sqsum = 0.0
        for p in self.master_params:
            sqsum += (p.grad ** 2).sum().item()
        
    def zero_grad(self, model_params):
        for param in model_params:
            # Taken from https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer.add_param_group
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()
