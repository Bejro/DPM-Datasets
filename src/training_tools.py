import os
from pathlib import Path
from typing import Optional, Union, List, Any

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
import torchvision
import torch
import torch.nn as nn
from torch import optim

from src.diffusion import Diffusion
from src.models import EMA, Autoencoder
from scripts.configs import TrainConfig
from src.utils import ImageLoader

CONFIG = TrainConfig
SAVE_DIR = Path(__file__).parent.parent / 'src/checkpoints'
RESULT_DIR = Path(__file__).parent.parent / 'results'


def _diffuse_and_step(diffusion: Diffusion, model: Autoencoder, optimizer: optim, images, labels) -> torch.Tensor:

    a = diffusion.sample_angles(images.shape[0]).to(model.device)
    if CONFIG.use_blur:
        images = diffusion.blur(images, 1 - (a - 3) / 177).real

    mse = nn.MSELoss()
    img_noised, noise = diffusion.noise_images(images, a)

    predicted_noise = model.unet(img_noised, a, labels)
    loss = mse(noise, predicted_noise)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def train_modules(
        diffusion: Diffusion, model: Autoencoder, optimizer: optim,
        images: torch.Tensor, labels: Optional[torch.Tenosr] = None,
        encoder: bool = True, decoder: bool = True
) -> torch.Tensor:
    def _train_on_batch(
            diffusion: Diffusion, model: Autoencoder, optimizer: optim,
            images: torch.Tensor, labels: Optional[Any] = None
    ):
        device = model.device
        images = torchvision.transforms.Normalize(diffusion.mean, diffusion.std)(images)
        images = images.to(device)

        if np.random.random() < 0.1:
            labels = torch.randn(len(images), 512).to(device) / 3
        else:
            labels = model.classif(images) + (
                torch.randn(len(images), 512).to(device) / 100 if np.random.random() < 0.5 else 0)
        return _diffuse_and_step(diffusion, model, optimizer, images, labels)

    def _train_u_on_batch(
            diffusion: Diffusion, model: Autoencoder, optimizer: optim,
            images: torch.Tensor, labels: torch.Tenosr
    ):
        device = model.device
        images = torchvision.transforms.Normalize(diffusion.mean, diffusion.std)(images)
        images = images.to(device)
        labels = labels.to(device)

        if np.random.random() < 0.1:
            labels = torch.randn(len(images), 512).to(device) / 3
        else:
            labels = labels + (torch.randn(len(images), 512).to(device) / 100 if np.random.random() < 0.5 else 0)
        return _diffuse_and_step(diffusion, model, optimizer, images, labels)

    def _train_c_on_batch(
            diffusion: Diffusion, model: Autoencoder, optimizer: optim,
            images: torch.Tensor, labels: torch.Tenosr
    ):
        device = model.device
        mse = nn.MSELoss()
        images = torchvision.transforms.Normalize(diffusion.mean, diffusion.std)(images)
        images = images.to(device)
        labels = labels.to(device)

        pred = model.classif(images)
        loss = mse(labels, pred)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    if encoder and decoder:
        return _train_on_batch(diffusion, model, optimizer, images)
    elif encoder:
        assert labels is not None
        return _train_c_on_batch(diffusion, model, optimizer, images, labels)
    else:
        assert labels is not None
        return _train_u_on_batch(diffusion, model, optimizer, images, labels)


def train(
        diffusion: Diffusion, model: Autoencoder, loader: ImageLoader, opt: optim, epochs: int,
        encoder: bool = True, decoder: bool = True,
        use_ema=False, is_supervised=False, logging_fc=None
):
    losses = []
    pbar = tqdm(loader)

    ema = None
    ema_model = None
    if use_ema:
        ema = EMA(0.995)
        ema_model = deepcopy(model).eval().requires_grad_(False)

    for epoch in range(epochs):
        for i, batch in enumerate(pbar):

            if not is_supervised:
                batch = [batch]

            if len(batch[0]) == 0:
                continue

            loss = train_modules(diffusion, model, opt, *batch, encoder=encoder, decoder=decoder)
            losses.append(loss)

            if ema is not None:
                ema.step_ema(ema_model, model)

            pbar.set_postfix(MSE=loss)

        if logging_fc is not None:
            logging_fc(model, ema_model, epoch, losses)


def log_reco_results(
        diffusion: Diffusion, save_name: str, imgs: torch.Tensor, model: Autoencoder, ema_model: Autoencoder,
        epoch: int, losses: List[Union[float, torch.Tensor]]
) -> None:
    save_dir = SAVE_DIR / CONFIG.experiment_name
    results_dir = RESULT_DIR / CONFIG.experiment_name
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    if ema_model is not None:
        model = ema_model

    torch.save(model.state_dict(), save_dir / '{save_name}_e{epoch}')

    imgs_n = torchvision.transforms.Normalize(diffusion.mean, diffusion.std)(imgs)
    n = len(imgs)

    sampled_images = diffusion.sample(model, n=n, angle_step=2, images=imgs_n.to(model.device)).clip(*CONFIG.img_window)
    _, axs = plt.subplots(2, n, figsize=(2 * n, 4), sharex='all')
    for i, ax in enumerate(axs.T):
        ax[1].imshow(sampled_images[i, 0], cmap='gray', vmin=-1000, vmax=1000)
        ax[0].imshow(imgs[i, 0], cmap='gray', vmin=-1000, vmax=1000)
        ax[0].axis("off")
        ax[1].axis("off")
    plt.tight_layout()
    plt.savefig(results_dir / f'{save_name}_reconstruction_e{epoch}.png')
    plt.close()

    np.save(results_dir / f'{save_name}_loss_e{epoch}.npy', losses)
