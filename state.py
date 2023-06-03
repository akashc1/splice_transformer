from dataclasses import dataclass
from typing import Callable
from pathlib import Path

from flax.training import train_state, checkpoints
from flax.core import frozen_dict


class TrainStateWithBN(train_state.TrainState):
    """
    In jax batch statistics are handled separately, so need to create new state type
    which also tracks the batch stats.
    """
    batch_stats: frozen_dict.FrozenDict


@dataclass
class ModelState:
    """
    Container for the state needed to forward a model. Used to run forward pass
    without creating a gradient transformation or optimizer state (as is needed in `TrainState`)
    """

    params: dict
    batch_stats: dict
    apply_fn: Callable

    @classmethod
    def from_ckpt_dir(cls, ckpt_dir, apply_fn):
        ckpt_dir = Path(ckpt_dir)
        if not ckpt_dir.name == 'checkpoints':
            print(f"Modifying {ckpt_dir=} to {(ckpt_dir := ckpt_dir / 'checkpoints')}")

        all_params = checkpoints.restore_checkpoint(ckpt_dir, None)
        return cls(all_params['params'], all_params['batch_stats'], apply_fn)

    @property
    def param_dict(self):
        return {'params': self.params, 'batch_stats': self.batch_stats}
