import argparse
import math
import os
from pathlib import Path
from typing import List

import numpy as np
import torch
import torchvision

from configs import InferenceConfig
from src.diffusion import Diffusion
from src.models import Autoencoder
from src.utils import load_data, generate_images, RejectionSampling

RESULT_DIR = Path(__file__).parent.parent / 'results'
CONFIG = InferenceConfig(Path(), Path())

if CONFIG.device_id is not None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(CONFIG.device_id)


def load_model(config: InferenceConfig) -> Autoencoder:
    model = Autoencoder(256, [64, 128, 256, 512, 1024], [32, 64, 128, 256, 512], 3, 512, config.device)
    model.load_state_dict(torch.load(config.state_dict_256_path))
    model.eval()
    return model


def save_images(
        config: InferenceConfig,
        negative_images: List[np.ndarray], positive_images: List[np.ndarray],
        output_dir: Path = RESULT_DIR
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    def save_images_in_batches(images, file_prefix):
        num_images = len(images)
        num_files = math.ceil(num_images / config.imgs_per_file)

        for i in range(num_files):
            start_idx = i * config.imgs_per_file
            end_idx = min((i + 1) * config.imgs_per_file, num_images)
            batch = images[start_idx:end_idx]
            torch.save(batch, output_dir / f'{file_prefix}_{i}.pt')

    save_images_in_batches(negative_images, 'negative_images')
    save_images_in_batches(positive_images,  'positive_images')

    print(f"Images saved in {output_dir}")

    sample_negative = torch.tensor(negative_images[:16])
    sample_positive = torch.tensor(positive_images[:16])

    torchvision.utils.save_image(sample_negative, output_dir / 'sample_negative_images.png', nrow=4)
    torchvision.utils.save_image(sample_positive, output_dir / 'sample_positive_images.png', nrow=4)


def main():
    parser = argparse.ArgumentParser(description="Inference script for sampling artificial images")
    parser.add_argument('--ds_path', type=Path, required=True, help="Path to the dataset")
    parser.add_argument('--ds_meta', type=Path, required=True, help="Path to the dataset metadata")

    args = parser.parse_args()
    config = InferenceConfig(**vars(args))

    if config.device_id is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = config.device_id

    images, labels = load_data(
        config.ds_path, config.ds_meta, config.ds_id_first, config.ds_id_last, config.img_res, 256
    )
    model = load_model(config)
    diffusion = Diffusion(img_size=256, device=config.device, lat_size=512, blurring=config.use_blur)
    diffusion.std = images.std(dim=(0, 2, 3))
    diffusion.mean = images.mean(dim=(0, 2, 3))

    training_vectors = torch.load(RESULT_DIR / 'training_vectors.pt')
    sampler = RejectionSampling(training_vectors, config.sample_threshold)

    negative_images, positive_images = generate_images(config, model, diffusion, images, labels, sampler=sampler)
    save_images(config, negative_images, positive_images)


if __name__ == "__main__":
    main()
