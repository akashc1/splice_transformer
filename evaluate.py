from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Callable, Dict, List

from absl import app, flags, logging
import chex
import colorama
import flax
import jax
from jax import numpy as jnp
from ml_collections import config_flags
from more_itertools import one
import numpy as np
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from constants import CONTEXT_LENGTHS, SEQUENCE_LENGTH, TEST_DATA_PATH
from dataset import H5SpliceDataset, get_test_dataset
from models import get_model

Fore = colorama.Fore
Style = colorama.Style


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def top_k_accuracy(logits, labels, ks=(0.5, 1, 2, 4), do_softmax=True):
    """
    Top-k accuracy as described in the original paper.
    Note that this implementation is actually different from theirs, since theirs has
    several bugs.
    """
    chex.assert_equal_shape([logits, labels])
    assert logits.ndim == 3, f'Expect 3D logits (B, T, C), but got {logits.shape}'
    assert labels.ndim == 3, f'Expect 3D labels (B, T, C), but got {labels.shape}'

    B, T, C = logits.shape
    logits, labels = logits.reshape(B * T, C), labels.reshape(B * T, C)

    probs = jax.nn.softmax(logits, axis=-1) if do_softmax else logits
    acceptor_probs, donor_probs = probs[:, 1], probs[:, 2]
    acceptor_labels, donor_labels = labels[:, 1], labels[:, 2]

    results = {}
    for name, (p, l) in zip(
        ('acceptor', 'donor'),
        (
            (acceptor_probs, acceptor_labels),
            (donor_probs, donor_labels),
        )
    ):
        true_idx = jnp.nonzero(l)[0]
        n_true = len(true_idx)

        if n_true == 0 or len(p) == 0:
            continue

        sorted_p_idx = jnp.argsort(p)

        for k in ks:
            top_p_idx = sorted_p_idx[-int(k * n_true):]
            den = min(n_true, len(top_p_idx)) + 1e-6
            acc = jnp.size(jnp.intersect1d(top_p_idx, true_idx)) / den
            thresh = p[sorted_p_idx[-int(k * n_true)]]

            results.update({
                f'{name}_{k}_accuracy': acc,
                f'{name}_{k}_threshold': float(thresh),
                f'{name}_num_true': n_true,
            })

        results[f'{name}_auc'] = average_precision_score(l, p)

    return results


def update_avg_stats(agg_stats: Dict[str, AverageMeter], new_stats: dict):
    for k, v in new_stats.items():
        if k.endswith('num_true'):
            continue

        splice_type = k.rsplit('_', 2)[0]
        n = new_stats[f'{splice_type}_num_true']
        agg_stats[k].update(v, n=n)


def fwd_batch(fwd_params, state, inputs):
    """
    Forward a single batch with no gradients
    """
    logits = state.apply_fn(
        fwd_params,
        inputs,
    )

    return logits


def batched_fwd(X, batch_size: int, state):
    """
    Forward a whole chunk of data (e.g. large segment of a chromosome)
    """
    p_fwd_fn = jax.pmap(fwd_batch, axis_name='batch', donate_argnums=2)

    world_size = jax.device_count()
    assert batch_size % world_size == 0, f"{batch_size=} must be divisible by {world_size=}"
    batch_size_per_device = batch_size // world_size
    shape_prefix = (world_size, batch_size_per_device)
    out = []

    fwd_params = {'params': state.params}
    if state.batch_stats is not None:
        fwd_params['batch_stats'] = state.batch_stats

    # data parallel full batches
    num_full_batches, num_ragged = divmod(X.shape[0], batch_size)
    for i in range(num_full_batches):
        Xb = X[i * batch_size:(i + 1) * batch_size]
        Xb = Xb.reshape(shape_prefix + Xb.shape[1:])

        out_b = p_fwd_fn(fwd_params, state, Xb).reshape((batch_size, SEQUENCE_LENGTH, -1))
        out.append(np.array(out_b))

    if not num_ragged:
        return np.concatenate(out)

    # data parallel ragged
    ragged_pmap, ragged_remaining = divmod(num_ragged, world_size)

    if ragged_pmap > 0:
        Xb = (
            X[num_full_batches * batch_size:-ragged_remaining]
            if ragged_remaining > 0 else X[num_full_batches * batch_size:]
        )
        Xb = Xb.reshape((world_size, ragged_pmap) + X.shape[1:])
        out_b = p_fwd_fn(fwd_params, state, Xb).reshape(
            (ragged_pmap * world_size, SEQUENCE_LENGTH, -1)
        )
        out.append(np.array(out_b))

    if not ragged_remaining:
        return np.concatenate(out)

    # final ragged, just one device
    Xb = X[-ragged_remaining:]
    out_b = fwd_batch(flax.jax_utils.unreplicate(fwd_params), flax.jax_utils.unreplicate(state), Xb)
    out.append(np.array(out_b))

    return np.concatenate(out)


def eval_dataset(ds: H5SpliceDataset, batch_size: int, state):
    """
    Run evaluation on all examples in a dataloader, parallelized across devices.

    Parameters
    ----------
    dl: DataLoader for the dataset we wish to eval. Assumes batch_size is a multiple of
        the number of devices.
    state: TrainState object with parameters, apply_fn etc. Assumes it's already duplicated
           across devices.

    Returns
    -------
    dict: aggregated statistics on the whole dataset
    """

    world_size = jax.device_count()
    assert batch_size % world_size == 0, f"{batch_size=} must be divisible by {world_size=}"

    agg_results = defaultdict(AverageMeter)
    for i in tqdm(range(len(ds)), desc='Evaluating dataset'):
        batch = ds[i]
        X_chunk, label_chunk = batch['x'], batch['y']

        logits_chunk = batched_fwd(X_chunk, batch_size, state)
        batch_results = top_k_accuracy(logits_chunk, label_chunk)
        update_avg_stats(agg_results, batch_results)

    return jax.tree_map(lambda x: x.avg, agg_results)


def print_accuracy_results(results: dict):

    acceptor_results = {k: v for k, v in results.items() if 'acceptor' in k}
    donor_results = {k: v for k, v in results.items() if 'donor' in k}

    logging.info(
        Fore.GREEN + Style.BRIGHT
        + f"Acceptor results:\n{Style.RESET_ALL}\n"
        + json.dumps(acceptor_results, indent=4)
    )
    logging.info(
        Fore.GREEN + Style.BRIGHT
        + f"Donor results:\n{Style.RESET_ALL}\n"
        + json.dumps(donor_results, indent=4)
    )


def get_models(
    basedir: Path,
    ckpt_prefix: str,
    apply_fn: Callable,
    num_models: int = 5,
):
    from state import ModelState
    dirs = sorted(basedir.glob(f'{ckpt_prefix}*'))[:num_models]
    assert (
        len(dirs) == num_models
    ), f'Expected {num_models} models with {ckpt_prefix=} in {basedir}, but only found {dirs}'
    model_states = [ModelState.from_ckpt_dir(d, apply_fn) for d in dirs]
    return model_states


def test(argv):
    del argv  # unused, but required to accept from absl.app

    config = FLAGS.config
    np.random.seed(config.seed)

    logging.info(f"Running evaluation on test set with context length {config.context_length}")
    assert (
        config.context_length in CONTEXT_LENGTHS
    ), f'{config.context_length=} not permitted, must be one of {CONTEXT_LENGTHS}'
    assert FLAGS.num_models <= 5, f'Expected a maximum of 5 models but got {FLAGS.num_models}!'

    world_size = jax.device_count()
    assert (
        config.batch_size % world_size == 0
    ), f"{config.batch_size} must be divisible by {world_size=}"

    test_ds = get_test_dataset(TEST_DATA_PATH, config.context_length)
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size if config.context_length == 10000 else 1,
        num_workers=config.num_workers,
        drop_last=True,
        shuffle=False,
    )

    # load the different models we're ensembling
    model = get_model(config)
    model_params = get_models(
        Path(FLAGS.ckpt_cache),
        FLAGS.ckpt_prefix,
        model.apply,
        FLAGS.num_models,
    )
    model_params = one(model_params)

    # measure parameter count before replicating
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(model_params))
    logging.info(f"Number of model parameters: {num_params:,}")

    model_params = flax.jax_utils.replicate(model_params)

    agg_results = defaultdict(AverageMeter)
    for batch in tqdm(test_loader, desc='Running evaluation on test dataset'):
        X_chunk, label_chunk = batch['x'].numpy(), batch['y'].numpy()
        if config.context_length < 10000:
            X_chunk, label_chunk = X_chunk.squeeze(), label_chunk.squeeze()

        logits = batched_fwd(X_chunk, config.batch_size, model_params)

        # run with chunk results, don't repeat softmax
        chunk_results = top_k_accuracy(logits, label_chunk)
        update_avg_stats(agg_results, chunk_results)
        del batch, X_chunk, label_chunk, logits, chunk_results

    # reduce to true averages for each value
    agg_results = jax.tree_map(lambda x: x.avg, agg_results)

    print_accuracy_results(agg_results)


if __name__ == '__main__':
    print(f"{jax.device_count()=}")
    print(f"Cmd: `python {' '.join(sys.argv)}`")

    FLAGS = flags.FLAGS

    flags.DEFINE_string('ckpt_cache', 'checkpoints', 'Root directory model data is stored.')
    flags.DEFINE_string(
        'ckpt_prefix',
        None,
        'Template of directory names containing models',
        required=True,
    )
    flags.DEFINE_integer(
        'num_models',
        None,
        'Number of models matching template to use',
        required=True,
    )
    config_flags.DEFINE_config_file(
        'config',
        None,
        'File path to the training hyperparameter configuration.',
        lock_config=True,
    )

    jax.config.config_with_absl()
    app.run(test)
