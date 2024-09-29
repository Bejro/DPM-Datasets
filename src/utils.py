import csv
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple, Union, Any, List
from zipfile import ZipFile
import pydicom as dicom

import numpy as np
import torch
import torchvision
from scipy import stats
from torch import nn
from torch.utils.data import Sampler
from tqdm import tqdm
import torch.nn.functional as F
import faiss

from scripts.configs import InferenceConfig
from src.diffusion import Diffusion
from src.models import Autoencoder

NUM_WORKERS = 4


def get_path(x: List[str]) -> str:
    return 'train/' + '/'.join(x) + '.dcm'


def get_arr(
        archive: ZipFile, path: str, window: Tuple[int, int] = (-1000, 3000)
) -> np.ndarray:
    imgdata = archive.open(path)
    dcm = dicom.dcmread(imgdata)
    arr = dcm.RescaleSlope * dcm.pixel_array + dcm.RescaleIntercept
    return arr.clip(*window)


def load_data(
        ds_path: Union[Path, str], ds_meta: Union[Path, str], begin: int = None, end: int = None,
        org_res: int = None, return_res: int = None
) -> (torch.Tensor, torch.Tensor):

    archive = zipfile.ZipFile(ds_path, 'r')
    x_path = []
    labels = []

    def process_image(idx: int, archive: ZipFile, x_path: List[List[str]], labels: List[int]) -> (torch.Tensor, int):
        path = get_path(x_path[idx])
        img = get_arr(archive, path)
        assert img.shape == (512, 512)
        img_tensor = torch.tensor(img)
        label = labels[idx]
        return img_tensor, label

    with open(ds_meta, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in tqdm(reader):
            x_path.append(row[:3])
            labels.append(int(row[3]))

    ds_x = torch.zeros(end - begin, org_res, org_res)
    ds_y = torch.zeros(end - begin)

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(process_image, i, archive, x_path, labels) for i in range(begin, end)]

    for i, future in tqdm(enumerate(as_completed(futures)), desc="Loading images", total=end - begin):
        img_tensor, label = future.result()
        ds_x[i] = img_tensor
        ds_y[i] = label

    ds_scaled = F.interpolate(ds_x.unsqueeze(1), size=return_res, mode='nearest')
    return ds_scaled, ds_y


def get_codes(
        model: Autoencoder, diffusion: Diffusion, ds: torch.Tensor, batch_size: int
) -> torch.Tensor:
    dataloader = ImageLoader(ds, batch_size=batch_size)
    codes = torch.zeros(len(ds), 512)
    c = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            img = batch[0]
            if len(img) == 0:
                continue
            img = torchvision.transforms.Normalize(diffusion.mean, diffusion.std)(img).to(model.device)
            code = model.classif(img)
            codes[c:c + len(code)] = code
            c += len(code)

    return codes


def generate_images(
        config: InferenceConfig, model: Autoencoder, diffusion: Diffusion, ds: torch.Tensor, ds_y: torch.Tensor,
        sampler: Optional['RejectionSampling'] = None
) -> (torch.Tensor, torch.Tensor):
    codes = get_codes(model, diffusion, ds, config.batch_256)

    codes_0 = codes[torch.nonzero((ds_y == 0)).squeeze()]
    codes_1 = codes[torch.nonzero((ds_y == 1)).squeeze()]

    codes_0_np = codes_0.cpu().numpy()
    codes_1_np = codes_1.cpu().numpy()
    kde_0 = stats.gaussian_kde(codes_0_np.T, bw_method=config.kde_bandwidth)
    kde_1 = stats.gaussian_kde(codes_1_np.T, bw_method=config.kde_bandwidth)
    gen_batch_size = config.generation_batch_size

    def sample_images(kde: stats.gaussian_kde, num_images: int) -> torch.Tensor:
        images = torch.zeros(num_images, 1, 256, 256)

        for i in tqdm(range(0, num_images, gen_batch_size), desc='Generating images'):
            n_to_gen = min(gen_batch_size, num_images - i)
            labels = torch.tensor(kde.resample(gen_batch_size * 10).T).to(config.device)
            if sampler is not None:
                labels = sampler.reject_samples(labels, n_to_gen)
                assert len(labels) == n_to_gen, 'Failed to sample enough latent codes, check the threshold.'
            else:
                labels = labels[:n_to_gen]
            new_generated = diffusion.sample(
                model, n=n_to_gen, cfg_scale=config.cfg_scale, angle_step=config.sampling_steps,
                labels=labels, noise_retention=config.noise_retention_share
            )
            images[i:i + gen_batch_size] = new_generated
        return images

    negative_images = sample_images(kde=kde_0, num_images=config.num_negative_images)
    positive_images = sample_images(kde=kde_1, num_images=config.num_positive_images)

    return negative_images, positive_images


class ImageLoader:

    def __init__(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None, batch_size: int = 64, transform: nn.Module = None):
        self.x = x
        self.transform = transform
        self.labels = labels
        self.i = 0
        self.bs = batch_size

    def __len__(self) -> int:
        return int(np.ceil(len(self.x) / self.bs))

    def __getitem__(self, index: Any) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:

        image = self.x[index]
        if self.transform is not None:
            image = self.transform(image)
        if self.labels is not None:
            return image, self.labels[index]
        else:
            return image

    def __iter__(self) -> 'ImageLoader':
        return self

    def __next__(self) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        res = self.__getitem__(np.s_[self.i * self.bs: (self.i + 1) * self.bs])
        if self.i > self.__len__() or len(res) == 0:
            self.i = 0
            raise StopIteration
        self.i += 1
        return res


class RejectionSampling:
    def __init__(self, vectors: torch.Tensor, th: int = 0.5):
        self.vectors = vectors
        self.index = faiss.IndexFlat(vectors.shape[1])
        self.index.add(vectors)
        self.th_norm = th ** 2 * 512

    def reject_samples(self, query: torch.Tensor, cap: int):
        res = query[(self.index.search(query, 1)[0])[:, 0] > self.th_norm]
        return res[:cap], len(res) / query.shape[0]
