from pathlib import Path
from typing import Optional, List, Tuple, Callable

from tqdm import tqdm
from copy import deepcopy
import torchvision
import torch
import torch.nn as nn
from torch import optim

from src.models import ClassifierWrap, Classifier
from scripts.configs import TrainConfig
from src.utils import ImageLoader

CONFIG = TrainConfig
SAVE_DIR = Path(__file__).parent.parent / 'src/checkpoints'
RESULT_DIR = Path(__file__).parent.parent / 'results'


def make_classifier() -> ClassifierWrap:
    px = 256
    classifier_core = Classifier(px, [32, 64, 128, 256, 512], 3, c_in=1, n_out=512).to(CONFIG.device)
    model = ClassifierWrap(classifier_core, 512, 2).to(CONFIG.device)
    return model


def accuracy(pred_logit: torch.Tensor, y: torch.Tensor) -> float:
    pred = pred_logit.argmax(1)
    return len(torch.nonzero(pred == y)) / len(y)


def validation_step(model: ClassifierWrap, ds: torch.Tensor, y: torch.Tensor) -> (float, float):
    ds = ds.clip(-600, 1000)
    ds_norm = torchvision.transforms.Normalize(-495., 500.)(ds).to(model.device)
    y = y.to(model.device)
    dataloader = ImageLoader(ds_norm, batch_size=64)

    with torch.no_grad():
        loss_fc = nn.CrossEntropyLoss(weight=torch.tensor([0.25, 0.75]).cuda())

        model = model.eval()

        pred = torch.tensor([]).cuda()
        for x in dataloader:
            p = model(x)
            pred = torch.cat([pred, p])

        model = model.train()

        acc = accuracy(pred, y.long())
        loss = loss_fc(pred, y.long())

    return acc, loss.item()


def train_new_model(
        data: Tuple[torch.Tensor, torch.Tensor],
        val: Tuple[torch.Tensor, torch.Tensor],
        model: Optional[ClassifierWrap] = None,
        model_gen: Optional[Callable[[], ClassifierWrap]] = None,
        transforms: nn.Module = nn.Identity(),
        epochs: int = 5,
        weight: torch.Tensor = torch.tensor([0.25, 0.75]).cuda(),
        saver: Optional['BestSaver'] = None
) -> (ClassifierWrap, List[float], List[float]):
    if model is None:
        model = model_gen()

    losses_ot = []
    losses_ov = []

    acc_ot = []
    acc_ov = []

    for epoch in range(epochs):
        dataloader = ImageLoader(data[0], data[1], batch_size=64, transform=transforms)
        lr = 1e-4 / (epoch + 1)
        opt = optim.AdamW(model.parameters(), lr=lr)
        print(f'lr = {lr}')
        pbar = tqdm(dataloader)
        loss_fc = nn.CrossEntropyLoss(weight=weight)
        acc_v, loss_v = 0, 0

        for i, batch in enumerate(pbar):

            images, labels = batch
            if len(images) == 0:
                continue

            images = images.clip(-600, 1000)

            images = torchvision.transforms.Normalize(-495., 500.)(images).to(model.device)
            labels = labels.to(model.device)

            pred = model(images)

            acc = accuracy(pred, labels.long())
            loss = loss_fc(pred, labels.long())

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses_ot.append(loss.item())
            acc_ot.append(acc)

            if i % 50 == 0:
                acc_v, loss_v = validation_step(model, *val)

                if saver is not None:
                    saver.save(model, acc_v)

                losses_ov.append(loss_v)
                acc_ov.append(acc_v)

            pbar.set_postfix(BCE=loss.item(), acc=acc, BCE_val=loss_v, acc_val=acc_v)

    return model, acc_ov, losses_ov


class BestSaver:
    def __init__(self):
        self.score = 0
        self.best_model = None

    def save(self, model: ClassifierWrap, score: float):
        if score > self.score:
            self.best_model = deepcopy(model)
            self.score = score
