import numpy as np
from more_itertools import zip_equal
import jax
from typing import Union
from flax import linen as nn


CONV_MODEL_CONFIG = {
    80: {
        'kernel_sizes': 11 * np.ones(4, dtype=int),
        'dilations': np.ones(4, dtype=int),
    },
    400: {
        'kernel_sizes': 11 * np.ones(8, dtype=int),
        'dilations': np.concatenate((np.ones(4), 4 * np.ones(4))),
    },
    2000: {
        'kernel_sizes': np.concatenate((11 * np.ones(8), 21 * np.ones(4))),
        'dilations': np.concatenate((np.ones(4), 4 * np.ones(4), 10 * np.ones(4))),
    },
    10000: {
        'kernel_sizes': np.concatenate((11 * np.ones(8), 21 * np.ones(4), 41 * np.ones(4))),
        'dilations': np.concatenate((np.ones(4), 4 * np.ones(4), 10 * np.ones(4), 25 * np.ones(4))),
    },
}


class ResidualUnit(nn.Module):

    dim: int
    k: int
    dilation: int

    def setup(self):
        self.net = nn.Sequential([
            nn.BatchNorm(),
            nn.relu,
            nn.Conv(self.dim, self.k, kernel_dilation=self.dilation),
            nn.BatchNorm(),
            nn.relu,
            nn.Conv(self.dim, self.k, kernel_dilation=self.dilation),
            nn.relu,
        ])

    def __call__(self, x):
        return x + self.net(x)


class DilatedConvSplicePredictor(nn.Module):

    dim: int
    kernel_sizes: Union[list, np.ndarray]
    dilations:  Union[list, np.ndarray]

    def __post_init__(self):
        assert len(self.kernel_sizes) == len(self.dilations)
        self.context_length = 2 * np.sum((self.kernel_sizes - 1) * self.dilations)
        super().__post_init__()

    @nn.compact
    def __call__(self, x):
        conv = nn.Conv(self.dim, 1, name='init_conv')(x)
        skip = nn.Conv(self.dim, 1, name='init_skip')(x)

        for i, (w, d) in enumerate(zip_equal(self.kernel_sizes, self.dilations)):
            conv = ResidualUnit(self.dim, w, d, name=f'residual{i}')(conv)

            if (i + 1) % 4 == 0 or i == len(self.kernel_sizes) - 1:
                dense = nn.Conv(self.dim, 1, name=f'dense{i // 4}')(conv)
                skip = skip + dense

        skip = skip[:, self.context_length // 2:-(self.context_length // 2), ...]
        return nn.Conv(3, 1, name='cls_final')(skip)


def get_conv_model(context_length: int):
    return DilatedConvSplicePredictor(dim=32, **CONV_MODEL_CONFIG[context_length])
