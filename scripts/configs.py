from pathlib import Path
from typing import Optional, Tuple

import attr


@attr.define(auto_attribs=True, frozen=True)
class Config:
    ds_meta: Path
    ds_path: Path
    device_id: Optional[str] = None
    device: str = 'cuda'
    img_window: Tuple[int, int] = (-1000, 3000)
    use_blur: bool = True
    state_dict_128_path: Path = Path(__file__).parent.parent / 'src/checkpoints/small_model_final.pt'
    state_dict_256_path: Path = Path(__file__).parent.parent / 'src/checkpoints/big_model_final.pt'
    img_res: int = 512
    batch_128: int = 32
    batch_256: int = 32


@attr.define(auto_attribs=True, frozen=True)
class TrainConfig(Config):
    experiment_name: str = "with_blur"
    ds_id_first: int = 0
    ds_id_last: int = 200000
    lr_128: int = 0.00001
    lr_256: int = 0.00001
    small_model_epochs: int = 5
    final_training_epochs: int = 2


@attr.define(auto_attribs=True)
class InferenceConfig(Config):
    ds_id_first: int = 400000
    ds_id_last: int = 600000
    num_negative_images: int = 20000
    num_positive_images: int = 10000
    cfg_scale: float = 2.8
    sampling_steps: int = 2
    kde_bandwidth: float = 0.07
    noise_retention_share: float = 0.65
    generation_batch_size: int = 100
    imgs_per_file: int = 5000
    sample_threshold: int = 0.41
