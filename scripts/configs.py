from pathlib import Path
from typing import Optional, Tuple

import attr


@attr.define(auto_attribs=True, frozen=True)
class TrainConfig:
    ds_meta: Path
    ds_path: Path
    device_id: Optional[str] = None
    device: str = 'cuda'
    img_window: Tuple[int, int] = (-1000, 3000)
    use_blur: bool = True
    experiment_name: str = "with_blur"
    ds_id_first: int = 0
    ds_id_last: int = 200000
    img_res: int = 512
    batch_128: int = 32
    batch_256: int = 32
    lr_128: int = 0.00001
    lr_256: int = 0.00001
    small_model_epochs: int = 5
    final_training_epochs: int = 2
    state_dict_128_path: Path = Path(__file__).parent.parent / 'src/checkpoints/small_model_final.pt'
