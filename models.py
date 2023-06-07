from dataclasses import fields
import functools
from typing import Union

from flax import linen as nn
from jax import numpy as jnp
from more_itertools import zip_equal
import numpy as np

from constants import SEQUENCE_LENGTH, ModelType

CONV_MODEL_CONFIG = {
    80: {
        'kernel_sizes': 11 * np.ones(4, dtype=int),
        'dilations': np.ones(4, dtype=int),
    },
    400: {
        'kernel_sizes': 11 * np.ones(8, dtype=int),
        'dilations': np.concatenate((np.ones(4, dtype=int), 4 * np.ones(4, dtype=int))),
    },
    2000: {
        'kernel_sizes': np.concatenate((11 * np.ones(8, dtype=int), 21 * np.ones(4, dtype=int))),
        'dilations': np.concatenate(
            (np.ones(4, dtype=int), 4 * np.ones(4, dtype=int), 10 * np.ones(4, dtype=int)),
        ),
    },
    10000: {
        'kernel_sizes': np.concatenate(
            (11 * np.ones(8, dtype=int), 21 * np.ones(4, dtype=int), 41 * np.ones(4, dtype=int))
        ),
        'dilations': np.concatenate(
            (
                np.ones(4, dtype=int),
                4 * np.ones(4, dtype=int),
                10 * np.ones(4, dtype=int),
                25 * np.ones(4, dtype=int)
            )
        ),
    },
}


class ResidualUnit(nn.Module):

    dim: int
    k: int
    dilation: int

    def setup(self):
        self.net = nn.Sequential([
            nn.BatchNorm(use_running_average=True, param_dtype=jnp.bfloat16),
            nn.relu,
            nn.Conv(self.dim, (self.k,), kernel_dilation=self.dilation, param_dtype=jnp.bfloat16),
            nn.BatchNorm(use_running_average=True),
            nn.relu,
            nn.Conv(self.dim, (self.k,), kernel_dilation=self.dilation, param_dtype=jnp.bfloat16),
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
        conv = nn.Conv(self.dim, (1,), param_dtype=jnp.bfloat16, name='init_conv')(x)
        skip = nn.Conv(self.dim, (1,), param_dtype=jnp.bfloat16, name='init_skip')(x)

        for i, (w, d) in enumerate(zip_equal(self.kernel_sizes, self.dilations)):
            conv = ResidualUnit(self.dim, w, (d,), name=f'residual{i}')(conv)

            if (i + 1) % 4 == 0 or i == len(self.kernel_sizes) - 1:
                dense = nn.Conv(
                    self.dim,
                    (1,),
                    param_dtype=jnp.bfloat16,
                    name=f'dense{i // 4}')(conv)
                skip = skip + dense

        skip = skip[:, self.context_length // 2:-(self.context_length // 2), ...]
        return nn.Conv(3, (1,), name='cls_final', param_dtype=jnp.bfloat16)(skip)


# use GPT initializations
Dense = functools.partial(
    nn.Dense,
    kernel_init=nn.initializers.normal(stddev=0.02, dtype=jnp.bfloat16),
    bias_init=nn.initializers.zeros,
    param_dtype=jnp.bfloat16,
)


class Block(nn.Module):
    emb_dim: int
    n_heads: int

    residual_dropout_prob: float
    attn_dropout_prob: float
    deterministic: bool

    n_blocks: int = 1  # for residual projection initialization

    def setup(self):
        self.attention = nn.SelfAttention(
            num_heads=self.n_heads,
            dropout_rate=self.attn_dropout_prob,
            deterministic=self.deterministic,
            param_dtype=jnp.bfloat16,
        )
        self.mlp = nn.Sequential(
            [
                Dense(4 * self.emb_dim),
                nn.gelu,
                nn.Dense(
                    self.emb_dim,
                    kernel_init=nn.initializers.normal(
                        stddev=0.02 / jnp.sqrt(2 * self.n_blocks),
                        dtype=jnp.bfloat16,
                    ),
                    bias_init=nn.initializers.zeros,
                ),
                nn.Dropout(
                    self.residual_dropout_prob, deterministic=self.deterministic
                ),
            ]
        )
        self.ln1 = nn.LayerNorm(param_dtype=jnp.bfloat16)
        self.ln2 = nn.LayerNorm(param_dtype=jnp.bfloat16)

    def __call__(self, x):
        B, T, _ = x.shape
        x = x.astype(jnp.bfloat16)
        x = x + self.attention(x)
        x = self.ln1(x)
        x = x + self.mlp(x)
        x = self.ln2(x)
        return x


class DownProjectBlock(Block):
    proj_length: int = SEQUENCE_LENGTH

    def setup(self):
        super().setup()

        self.attention = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            dropout_rate=self.attn_dropout_prob,
            deterministic=self.deterministic,
            param_dtype=jnp.bfloat16,
        )

    def __call__(self, x):
        B, T = x.shape[:2]
        x = x.astype(jnp.bfloat16)

        # query is center block of length proj_length
        trunc_length = T - self.proj_length
        xq = x[:, trunc_length // 2:-trunc_length // 2]
        x = x + self.attention(inputs_q=xq, inputs_kv=x, deterministic=self.deterministic)

        x = self.ln1(x)
        x = x + self.mlp(x)
        x = self.ln2(x)
        return x


class Transformer(nn.Module):
    context_length: int

    emb_dim: int

    n_blocks: int
    n_heads: int

    emb_dropout_prob: float
    block_dropout_prob: float
    attn_dropout_prob: float
    deterministic: bool

    perceiver: bool = False

    n_classes: int = 3

    def setup(self):
        self.token_emb = Dense(self.emb_dim)
        # TODO: try sinusoidal embedding initializer
        self.pos_embedding = self.param(
            'pos_embedding',
            nn.initializers.normal(stddev=1 / jnp.sqrt(self.emb_dim), dtype=jnp.bfloat16),
            (1, self.context_length + SEQUENCE_LENGTH, self.emb_dim),
        )
        self.dropout = nn.Dropout(
            self.emb_dropout_prob, deterministic=self.deterministic
        )

        blocks = []
        block_params = dict(
            emb_dim=self.emb_dim,
            n_heads=self.n_heads,
            n_blocks=self.n_blocks,  # for residual projection initialization
            attn_dropout_prob=self.attn_dropout_prob,
            residual_dropout_prob=self.block_dropout_prob,
            deterministic=self.deterministic,
        )
        if self.perceiver:
            block_params['proj_length'] = SEQUENCE_LENGTH
            blocks.append(DownProjectBlock(**block_params))
        else:
            blocks.append(Block(**block_params))
        blocks.extend(Block(**block_params) for _ in range(self.n_blocks - 1))
        self.transformer = nn.Sequential(blocks)

        self.ln = nn.LayerNorm(param_dtype=jnp.bfloat16)
        self.head = Dense(self.n_classes)

    def __call__(self, x):
        t = x.shape[1]  # (B, SEQUENCE_LENGTH + CONTEXT_LENGTH, C)

        x = x.astype(jnp.bfloat16)
        emb_tokens = self.token_emb(x)
        emb_pos = self.pos_embedding[:, :t, :]
        x = emb_tokens + emb_pos
        x = self.dropout(x)
        x = self.transformer(x)

        # perceiver squashes to SEQUENCE_LENGTH
        if not self.perceiver:
            x = x[:, self.context_length // 2:-self.context_length // 2]

        x = self.ln(x)
        x = self.head(x)
        return x


class Enformer(nn.Module):
    # meta params
    context_length: int

    # embedding dimension
    emb_dim: int

    # convolutional params
    kernel_sizes: Union[list, np.ndarray]
    dilations:  Union[list, np.ndarray]

    # transformer params
    n_blocks: int
    n_heads: int
    emb_dropout_prob: float
    block_dropout_prob: float
    attn_dropout_prob: float
    deterministic: bool

    perceiver: bool = False

    n_classes: int = 3

    @nn.compact
    def __call__(self, x):
        x = x.astype(jnp.bfloat16)
        conv = nn.Conv(self.emb_dim, (1,), name='init_conv', param_dtype=jnp.bfloat16)(x)
        skip = nn.Conv(self.emb_dim, (1,), name='init_skip', param_dtype=jnp.bfloat16)(x)

        for i, (w, d) in enumerate(zip_equal(self.kernel_sizes, self.dilations)):
            conv = ResidualUnit(self.emb_dim, w, (d,), name=f'residual{i}')(conv)

            if (i + 1) % 4 == 0 or i == len(self.kernel_sizes) - 1:
                dense = nn.Conv(
                    self.emb_dim,
                    (1,),
                    name=f'dense{i // 4}',
                    param_dtype=jnp.bfloat16,
                )(conv)
                skip = skip + dense

        out = Transformer(
            self.context_length,
            self.emb_dim,
            self.n_blocks,
            self.n_heads,
            self.emb_dropout_prob,
            self.block_dropout_prob,
            self.attn_dropout_prob,
            self.deterministic,
            perceiver=self.perceiver,
            name='transformer',
        )(skip)
        return out


def get_conv_model(context_length: int):
    return DilatedConvSplicePredictor(dim=32, **CONV_MODEL_CONFIG[context_length])


def get_bert(config):
    # skip fields from base `nn.Module`
    fnames = [f.name for f in fields(Transformer) if f.name not in {'parent', 'name'}]
    transformer_kwargs = {k: getattr(config, k) for k in fnames}
    return Transformer(**transformer_kwargs)


def get_enformer(config):
    xformer_fnames = [f.name for f in fields(Transformer) if f.name not in {'parent', 'name'}]
    xformer_kwargs = {k: getattr(config, k) for k in xformer_fnames}
    enformer_kwargs = {**xformer_kwargs, **CONV_MODEL_CONFIG[config.conv_setting]}
    return Enformer(**enformer_kwargs)


def get_model(config):
    return {
        ModelType.DILATED_CONV: lambda c: get_conv_model(c.context_length),
        ModelType.BERT: lambda c: get_bert(c),
        ModelType.ENFORMER: lambda c: get_enformer(c),
    }[config.model_type](config)
