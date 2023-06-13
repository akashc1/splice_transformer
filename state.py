from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from flax import struct
from flax.core import frozen_dict
from flax.training import checkpoints, train_state


class TrainStateWithBN(train_state.TrainState):
    """
    In jax batch statistics are handled separately, so need to create new state type
    which also tracks the batch stats.
    """
    batch_stats: frozen_dict.FrozenDict


@dataclass
class ModelState(struct.PyTreeNode):
    """
    Container for the state needed to forward a model. Used to run forward pass
    without creating a gradient transformation or optimizer state (as is needed in `TrainState`)
    """
    params: dict = struct.field(pytree_node=True)
    apply_fn: Callable = struct.field(pytree_node=False)

    # batch stats only used in models that use batchnorm
    batch_stats: Optional[dict] = struct.field(pytree_node=True, default=None)

    @classmethod
    def from_ckpt_dir(cls, ckpt_dir, apply_fn):
        ckpt_dir = Path(ckpt_dir)
        if not ckpt_dir.name == 'checkpoints':
            print(f"Modifying {ckpt_dir=} to {(ckpt_dir := ckpt_dir / 'checkpoints')}")

        all_params = checkpoints.restore_checkpoint(ckpt_dir, None, parallel=False)
        return cls(all_params['params'], apply_fn, all_params.get('batch_stats'))
