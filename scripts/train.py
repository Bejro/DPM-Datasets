import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import csv
from tqdm import tqdm
import torchvision
from torch import optim
import torch
import torch.nn.functional as F
import zipfile
from functools import partial

from scripts.training_tools import train, log_reco_results, get_path, get_arr
from src.utils import ImageLoader
from src.diffusion import Diffusion
from src.models import Autoencoder
from scripts.configs import TrainConfig

CONFIG = TrainConfig
DEVICE = CONFIG.device

if CONFIG.device_id is not None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = CONFIG.device_id


def load_data(
        begin: Optional[int] = None, end: Optional[int] = None, img_res: Optional[int] = None
) -> Tuple[torch.Tesnor, torch.Tensor]:
    begin = begin or CONFIG.ds_id_first
    end = end or CONFIG.ds_id_last
    img_res = img_res or CONFIG.img_res

    archive = zipfile.ZipFile(CONFIG.ds_path, 'r')
    x_path = []

    with open(CONFIG.ds_meta, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in tqdm(reader):
            x_path.append(row[:3])
    ds = torch.zeros(end - begin, img_res, img_res)
    for i, j in tqdm(enumerate(range(begin, end))):
        path = get_path(x_path[j])
        img = get_arr(archive, path)
        assert img.shape == (img_res, img_res)
        ds[i] = torch.tensor(img)

    ds128 = F.interpolate(ds.unsqueeze(1), size=128, mode='nearest')
    ds256 = F.interpolate(ds.unsqueeze(1), size=256, mode='nearest')

    return ds128, ds256


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


def parse_args():
    parser = argparse.ArgumentParser(description="Training script arguments")

    # Adding arguments that match TrainConfig
    parser.add_argument('--ds_meta', type=Path, required=True, help="Path to dataset metadata")
    parser.add_argument('--ds_path', type=Path, required=True, help="Path to dataset")
    parser.add_argument('--device_id', type=str, default=None, help="Device ID to use (Optional)")
    parser.add_argument('--device', type=str, default='cuda', help="Device to use (default: cuda)")
    parser.add_argument('--img_window', type=int, nargs=2, default=(-1000, 3000),
                        help="Image window for processing (default: (-1000, 3000))")
    parser.add_argument('--use_blur', type=bool, default=True,
                        help="Whether to use blur in preprocessing (default: True)")
    parser.add_argument('--experiment_name', type=str, default="with_blur",
                        help="Name of the experiment (default: with_blur)")
    parser.add_argument('--ds_id_first', type=int, default=0, help="First dataset ID (default: 0)")
    parser.add_argument('--ds_id_last', type=int, default=200000, help="Last dataset ID (default: 200000)")
    parser.add_argument('--img_res', type=int, default=512, help="Image resolution (default: 512)")
    parser.add_argument('--batch_128', type=int, default=32, help="Batch size for 128 resolution images (default: 32)")
    parser.add_argument('--batch_256', type=int, default=32, help="Batch size for 256 resolution images (default: 32)")
    parser.add_argument('--lr_128', type=float, default=0.00001,
                        help="Learning rate for 128 resolution images (default: 0.00001)")
    parser.add_argument('--lr_256', type=float, default=0.00001,
                        help="Learning rate for 256 resolution images (default: 0.00001)")
    parser.add_argument('--small_model_epochs', type=int, default=5,
                        help="Number of epochs for the small model (default: 5)")
    parser.add_argument('--final_training_epochs', type=int, default=2,
                        help="Number of epochs for final training (default: 2)")
    parser.add_argument('--state_dict_128_path', type=Path,
                        default=Path(__file__).parent.parent / 'src/checkpoints/small_model_final.pt',
                        help="Path to small model checkpoint (default: src/checkpoints/small_model_final.pt)")

    return parser.parse_args()


def main():
    global CONFIG
    args = parse_args()

    config = TrainConfig(
        ds_meta=args.ds_meta,
        ds_path=args.ds_path,
        device_id=args.device_id,
        device=args.device,
        img_window=(args.img_window[0], args.img_window[1]),
        use_blur=args.use_blur,
        experiment_name=args.experiment_name,
        ds_id_first=args.ds_id_first,
        ds_id_last=args.ds_id_last,
        img_res=args.img_res,
        batch_128=args.batch_128,
        batch_256=args.batch_256,
        lr_128=args.lr_128,
        lr_256=args.lr_256,
        small_model_epochs=args.small_model_epochs,
        final_training_epochs=args.final_training_epochs,
        state_dict_128_path=args.state_dict_128_path
    )

    print(config)
    CONFIG = config

    ds_128, ds_256 = load_data()
    model_small, model_big, diffusion = init_models(ds_128)
    train_big_model(diffusion, model_big, model_small, ds_128, ds_256)


if __name__ == "__main__":
    main()
