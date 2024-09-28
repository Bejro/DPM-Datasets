import csv
import zipfile
from pathlib import Path
from typing import Optional, Tuple, Union, Any, List
from zipfile import ZipFile

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F


def get_path(x: List[str]) -> Path:
    return Path('train/' + '/'.join(x) + '.dcm')


def get_arr(
        archive: ZipFile, path: Union[Path, str], dicom=None, window: Tuple[int, int] = (-1000, 3000)
) -> np.ndarray:
    imgdata = archive.open(path)
    dcm = dicom.dcmread(imgdata)
    arr = dcm.RescaleSlope * dcm.pixel_array + dcm.RescaleIntercept
    return arr.clip(*window)


def load_data(
        ds_path: Union[Path, str], ds_meta: Union[Path, str],  begin: int = None, end: int = None,
        org_res: int = None, return_res: int = None
) -> (torch.Tensor, torch.Tensor):

    archive = zipfile.ZipFile(ds_path, 'r')
    x_path = []
    labels = []

    with open(ds_meta, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in tqdm(reader):
            x_path.append(row[:3])
            labels.append(int(row[3]))
    ds_x = torch.zeros(end - begin, org_res, org_res)
    ds_y = torch.zeros(end - begin)
    for i, j in tqdm(enumerate(range(begin, end))):
        path = get_path(x_path[j])
        img = get_arr(archive, path)
        assert img.shape == (org_res, org_res)
        ds_x[i] = torch.tensor(img)
        ds_y[i] = labels[j]

    ds_scaled = F.interpolate(ds_x.unsqueeze(1), size=return_res, mode='nearest')
    return ds_scaled, ds_y


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
