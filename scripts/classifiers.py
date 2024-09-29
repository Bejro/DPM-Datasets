import argparse
from pathlib import Path

import torch
import torchvision

from scripts.configs import EvalConfig
from src.classifier_tools import BestSaver, make_classifier, train_new_model
from src.utils import load_data

RESULT_DIR = Path(__file__).parent.parent / 'results'


def get_new_shuffled_data(images, labels, num_pos, num_neg) -> (torch.Tensor, torch.Tensor):
    images_neg = images[torch.nonzero((labels == 0))]
    images_pos = images[torch.nonzero((labels == 1))]

    images_neg = images_neg[torch.randperm(images_neg.shape[0])]
    images_pos = images_pos[torch.randperm(images_pos.shape[0])]

    x_0 = images_neg[:num_neg]
    x_1 = images_pos[:num_pos]

    ds = torch.cat((x_0, x_1), 0)
    y = torch.zeros(len(ds))
    y[len(x_0):] = 1

    random = torch.randperm(len(ds))

    y = y[random]
    ds = ds[random]

    return ds.squeeze(1), y


def load_artificial_images(output_dir: Path = RESULT_DIR) -> (torch.Tensor, torch.Tensor):
    def load_images_from_batches(file_prefix: str) -> torch.Tensor:
        image_batches = []
        batch_idx = 0

        while True:
            file_path = output_dir / f'{file_prefix}_{batch_idx}.pt'
            if not file_path.exists():
                break
            batch = torch.load(file_path)
            image_batches.append(batch)
            batch_idx += 1

        return torch.cat(image_batches, dim=0)

    negative_images = load_images_from_batches('negative_images')
    positive_images = load_images_from_batches('positive_images')

    ds = torch.cat([negative_images, positive_images])
    y = torch.zeros(len(ds))
    y[len(negative_images):] = 1

    return ds, y


def main():
    parser = argparse.ArgumentParser(description="Inference script for sampling artificial images")
    parser.add_argument('--ds_path', type=Path, required=True, help="Path to the dataset")
    parser.add_argument('--ds_meta', type=Path, required=True, help="Path to the dataset metadata")

    args = parser.parse_args()
    config = EvalConfig(**vars(args))

    fake_images, fake_labels = load_artificial_images(args.ds_path)

    real_images, real_labels = load_data(
        config.ds_path, config.ds_meta, config.ds_id_first, config.ds_id_last, config.img_res, 256
    )

    val_images, val_labels = load_data(
        config.ds_path, config.ds_meta, config.val_id_first, config.val_id_last, config.img_res, 256
    )

    transforms = torch.nn.Sequential(
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.RandomResizedCrop(256, (0.85, 1), (0.95, 1.05)),
        torchvision.transforms.RandomApply(torch.nn.ModuleList([
            torchvision.transforms.Resize(290),
            torchvision.transforms.RandomRotation(15),
            torchvision.transforms.CenterCrop(256),
        ]), p=0.7)
    )

    loss_r, loss_f, loss_b = [], [], []
    acc_r, acc_f, acc_b = [], [], []
    sav_r, sav_f, sav_b = [], [], []

    for i in range(config.num_classifiers):
        xv, yv = get_new_shuffled_data(val_images, val_labels, 9000, 27000)
        xr, yr = get_new_shuffled_data(real_images, real_labels, 9000, 27000)
        xf, yf = get_new_shuffled_data(fake_images, fake_labels, 9000, 27000)
        xfb, yfb = get_new_shuffled_data(fake_images, fake_labels, 9000, 9000)
        xb, yb = get_new_shuffled_data(torch.cat([xfb, xr]), torch.cat([yfb, yr]), 18000, 18000)

        sav_r.append(BestSaver())
        sav_f.append(BestSaver())
        sav_b.append(BestSaver())

        _, acc, loss = train_new_model((xr, yr), (xv, yv), model_gen=make_classifier, epochs=5,
                                       saver=sav_r[-1], transforms=transforms)
        loss_r.append(loss)
        acc_r.append(acc)

        _, acc, loss = train_new_model((xf, yf), (xv, yv), model_gen=make_classifier, epochs=5,
                                       saver=sav_f[-1], transforms=transforms)
        loss_f.append(loss)
        acc_f.append(acc)

        _, acc, loss = train_new_model((xb, yb), (xv, yv), model_gen=make_classifier, epochs=5,
                                       weight=torch.tensor([1., 1.]).cuda(), saver=sav_b[-1], transforms=transforms)
        loss_b.append(loss)
        acc_b.append(acc)

        print(f"accuracies: real: {round(sav_r[-1].score, 3)}, fake: {round(sav_f[-1].score, 3)}, "
              f"balanced: {round(sav_b[-1].score, 3)}")
        print(f"saving models for data shuffle number{i}")
        torch.save(sav_r[i].best_model, f'classifiers_trained/a_r{i}')
        torch.save(sav_f[i].best_model, f'classifiers_trained/a_f{i}')
        torch.save(sav_b[i].best_model, f'classifiers_trained/a_b{i}')


if __name__ == "__main__":
    main()
