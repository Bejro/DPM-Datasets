from typing import Optional, Tuple, Union, Any

import numpy as np
import torch
from numpy.lib.index_tricks import IndexExpression
from torch import nn


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
