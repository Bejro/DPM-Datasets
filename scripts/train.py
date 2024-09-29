import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from tqdm import tqdm
import torchvision
from torch import optim
import torch
from functools import partial

from src.training_tools import train, log_reco_results
from src.utils import ImageLoader, load_data, get_codes
from src.diffusion import Diffusion
from src.models import Autoencoder
from scripts.configs import TrainConfig

CONFIG = TrainConfig
DEVICE = CONFIG.device
RESULT_DIR = Path(__file__).parent.parent / 'results'

if CONFIG.device_id is not None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(CONFIG.device_id)


def load_data_for_training(
        begin: Optional[int] = None, end: Optional[int] = None, org_res: Optional[int] = None
) -> Tuple[torch.Tensor, ...]:
    begin = begin or CONFIG.ds_id_first
    end = end or CONFIG.ds_id_last
    org_res = org_res or CONFIG.img_res

    res = [load_data(CONFIG.ds_path, CONFIG.ds_meta, begin, end, org_res, px)[0] for px in [128, 256]]
    return tuple(res)


def init_models(images: torch.Tensor) -> (Autoencoder, Autoencoder, Diffusion):

    model_128 = Autoencoder(128, [64, 128, 256, 512, 1024], [32, 64, 128, 256, 512], 3, 512, DEVICE)
    model_256 = Autoencoder(256, [64, 128, 256, 512, 1024], [32, 64, 128, 256, 512], 3, 512, DEVICE)

    diffusion = Diffusion(img_size=128, device=DEVICE, lat_size=512, blurring=CONFIG.use_blur)
    diffusion.std = images.std(dim=(0, 2, 3))
    diffusion.mean = images.mean(dim=(0, 2, 3))

    return model_128, model_256, diffusion


def prepare_or_train_small_model(
        diffusion: Diffusion, small_model: Autoencoder, dataloader: ImageLoader, vis_ex_imgs: torch.Tensor
) -> None:
    optimizer = optim.AdamW(small_model.parameters(), lr=CONFIG.lr_128)

    state_dict_128_path = CONFIG.state_dict_128_path
    if state_dict_128_path is not None:
        small_model.load_state_dict(torch.load(state_dict_128_path))
        print('Small model loaded from ckpt.')
    else:
        # model_128.load_state_dict(torch.load(state_dict_128_path))
        print('training small model...')

        train(
            diffusion, small_model, dataloader, optimizer, CONFIG.small_model_epochs,
            use_ema=True, is_supervised=False,
            logging_fc=partial(log_reco_results, 'small_model', vis_ex_imgs)
        )


def prepare_labels_dataloader(
        diffusion: Diffusion, small_model: Autoencoder, ds128: torch.Tensor, ds256: torch.Tensor
) -> ImageLoader:
    dataloader_imgs = ImageLoader(ds128, batch_size=CONFIG.batch_128)
    labels = []
    small_model.eval()
    with torch.no_grad():
        for imgs in tqdm(dataloader_imgs):
            imgs_n = torchvision.transforms.Normalize(diffusion.mean, diffusion.std)(imgs)
            label_batch = small_model.classif(imgs_n.to('cuda')).cpu().numpy()
            labels.extend(label_batch)

    labels_set = np.array(labels)
    labels_set = torch.Tensor(labels_set)

    rand_order = torch.randperm(len(ds256))
    ds_shuffled = ds256[rand_order]
    labels_shuffled = labels_set[rand_order]

    dataloader = ImageLoader(ds_shuffled, labels_shuffled, batch_size=CONFIG.batch_256)

    return dataloader


def train_big_model(
        diffusion: Diffusion, big_model: Autoencoder, small_model: Autoencoder, ds128: torch.Tensor, ds256: torch.Tensor
) -> None:
    rand_order = torch.randperm(len(ds256))
    progress_visualisation_examples = {"small": ds128[rand_order], "big": ds256[rand_order]}

    ds_shuffled = ds128[rand_order]
    dataloader_128 = ImageLoader(ds_shuffled, batch_size=CONFIG.batch_128)

    print('Small model preparation.')
    prepare_or_train_small_model(diffusion, small_model, dataloader_128, progress_visualisation_examples["small"])

    print('label_extraction...')
    supervised_dataloader = prepare_labels_dataloader(diffusion, small_model, ds128, ds256)

    print('training bigger model...')

    print('fitting DPM decoder...')
    optimizer = optim.AdamW(big_model.unet.parameters(), lr=CONFIG.lr_256)
    diffusion.img_size = 256
    train(
        diffusion, big_model, supervised_dataloader, optimizer, epochs=1,
        encoder=False, use_ema=True, is_supervised=True,
        logging_fc=partial(log_reco_results, save_name='big_fixed', imgs=progress_visualisation_examples["big"])
    )

    print('fitting semantic encoder...')
    optimizer = optim.AdamW(big_model.classif.parameters(), lr=CONFIG.lr_256)
    train(
        diffusion, big_model, supervised_dataloader, optimizer,  epochs=1,
        decoder=False, use_ema=False, is_supervised=True, logging_fc=None
    )

    print('end to end')

    optimizer = optim.AdamW(big_model.parameters(), lr=CONFIG.lr_256)
    dataloader_256 = ImageLoader(ds_shuffled, batch_size=CONFIG.batch_256)

    train(
        diffusion, big_model, dataloader_256, optimizer, epochs=CONFIG.final_training_epochs,
        use_ema=True, is_supervised=False,
        logging_fc=partial(log_reco_results, 'big_freed', progress_visualisation_examples["big"])
    )


def extract_codes(model: Autoencoder, diffusion: Diffusion, ds: torch.Tensor):
    codes = get_codes(model, diffusion, ds, CONFIG.batch_256)
    torch.save(codes, RESULT_DIR / 'training_codes.pt')


def main():
    global CONFIG
    parser = argparse.ArgumentParser(description="Inference script for sampling artificial images")
    parser.add_argument('--ds_path', type=Path, required=True, help="Path to the dataset")
    parser.add_argument('--ds_meta', type=Path, required=True, help="Path to the dataset metadata")

    args = parser.parse_args()
    config = TrainConfig(**vars(args))
    CONFIG = config

    ds_128, ds_256 = load_data_for_training()
    model_small, model_big, diffusion = init_models(ds_128)
    train_big_model(diffusion, model_big, model_small, ds_128, ds_256)
    print('Training finished, extracting codes for a later use...')
    extract_codes(model_big, diffusion, ds_256)


if __name__ == "__main__":
    main()
