from flax.training import train_state
from flax.core import frozen_dict


class TrainStateWithBN(train_state.TrainState):
    """
    In jax batch statistics are handled separately, so need to create new state type
    which also tracks the batch stats.
    """
    batch_stats: frozen_dict.FrozenDict

