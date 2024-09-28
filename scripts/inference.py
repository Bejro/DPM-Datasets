import argparse
import os
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
import torchvision
from tqdm import tqdm
from scipy import stats

from scripts.configs import InferenceConfig
from src.diffusion import Diffusion
from src.models import Autoencoder
from src.utils import load_data

RESULT_DIR = Path(__file__).parent.parent / 'results'


def load_model(config: InferenceConfig) -> Autoencoder:
    model = Autoencoder(256, [64, 128, 256, 512, 1024], [32, 64, 128, 256, 512], 3, 512, config.device)
    model.load_state_dict(torch.load(config.state_dict_256_path))
    model.eval()
    return model


def generate_images(
        config: InferenceConfig, model: Autoencoder, diffusion: Diffusion
) -> Tuple[List[np.ndarray], List[np.ndarray]]:

    ds, ds_y = load_data(config.ds_path, config.ds_meta, config.ds_id_first, config.ds_id_last, config.img_res, 256)

    codes = []
    batch_size = config.batch_256
    with torch.no_grad():
        for i in range(0, len(ds), batch_size):
            batch = ds[i:i + batch_size].to(config.device)
            codes.append(model.encode(batch))
    codes = torch.cat(codes, dim=0)

    codes_0 = codes[torch.nonzero((ds_y == 0)).squeeze()]
    codes_1 = codes[torch.nonzero((ds_y == 1)).squeeze()]

    codes_0_np = codes_0.cpu().numpy()
    codes_1_np = codes_1.cpu().numpy()
    kde_0 = stats.gaussian_kde(codes_0_np.T, bw_method=config.kde_bandwidth)
    kde_1 = stats.gaussian_kde(codes_1_np.T, bw_method=config.kde_bandwidth)

    def sample_images(kde: stats.gaussian_kde, num_images: int) -> List[np.ndarray]:
        images = []

        for _ in tqdm(range(0, num_images, batch_size)):

            idx = np.random.choice(len(ds), batch_size)
            batch = ds[idx].to(config.device)
            images.append(batch.cpu())

            labels = torch.tensor(kde.resample(batch_size).T).to(config.device)
            positive_batch = diffusion.sample(
                model, n=batch_size, cfg_scale=config.cfg_scale, angle_step=config.sampling_steps,
                labels=labels, noise_retention=config.noise_retention_share
            )
            images.extend(positive_batch.cpu())

        return images

    negative_images = sample_images(kde=kde_0, num_images=config.num_negative_images)
    positive_images = sample_images(kde=kde_1, num_images=config.num_positive_images)

    return negative_images, positive_images


def save_images(
        config: InferenceConfig,
        negative_images: List[np.ndarray], positive_images: List[np.ndarray],
        output_dir: Path = RESULT_DIR
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    torch.save(negative_images, output_dir / 'negative_images.pt')
    torch.save(positive_images, output_dir / 'positive_images.pt')

    print(f"Images saved in {output_dir}")

    sample_negative = torch.tensor(negative_images[:16])
    sample_positive = torch.tensor(positive_images[:16])

    torchvision.utils.save_image(sample_negative, output_dir / 'sample_negative_images.png', nrow=4, normalize=True,
                                 value_range=config.img_window)
    torchvision.utils.save_image(sample_positive, output_dir / 'sample_positive_images.png', nrow=4, normalize=True,
                                 value_range=config.img_window)


def main():
    parser = argparse.ArgumentParser(description="Inference script for sampling positive images")
    parser.add_argument('--model_path', type=Path, required=True, help="Path to the trained model")
    parser.add_argument('--ds_path', type=Path, required=True, help="Path to the dataset")
    parser.add_argument('--ds_meta', type=Path, required=True, help="Path to the dataset metadata")
    parser.add_argument('--output_dir', type=Path, required=True, help="Directory to save output images")
    parser.add_argument('--device', type=str, default='cuda', help="Device to use for inference")
    parser.add_argument('--num_images', type=int, default=10000, help="Number of images to generate")
    parser.add_argument('--cfg_scale', type=float, default=2.8, help="CFG scale for sampling")
    parser.add_argument('--sampling_steps', type=int, default=2, help="Number of sampling steps")
    parser.add_argument('--kde_bandwidth', type=float, default=0.1, help="Bandwidth for KDE")
    parser.add_argument('--label_share', type=float, default=0.65, help="Label share for sampling")

    args = parser.parse_args()
    config = InferenceConfig(**vars(args))

    if config.device_id is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = config.device_id

    model = load_model(config)
    diffusion = Diffusion(img_size=256, device=config.device, lat_size=512, blurring=config.use_blur)

    negative_images, positive_images = generate_images(config, model, diffusion)
    save_images(config, negative_images, positive_images)


if __name__ == "__main__":
    main()
