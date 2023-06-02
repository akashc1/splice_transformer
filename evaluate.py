from collections import defaultdict
import json
import sys

from flax.training import checkpoints
from absl import app, flags, logging
import chex
import colorama
import flax
import jax
from jax import numpy as jnp
from ml_collections import config_flags
import numpy as np
from sklearn.metrics import average_precision_score
from tqdm.auto import tqdm

from constants import CONTEXT_LENGTHS, SEQUENCE_LENGTH
from models import get_conv_model
from dataset import H5SpliceDataset, get_test_dataset
from state import TrainStateWithBN

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


def top_k_accuracy(logits, labels, ks=(0.5, 1, 2, 4)):
    """
    Top-k accuracy as described in the original paper.
    Note that this implementation is actually different from theirs, since theirs has
    several bugs.
    """
    chex.assert_equal_shape([logits, labels])
    assert logits.ndim == 3, f'Expect 3D logits (B, T, C), but got {logits.shape}'
    assert labels.ndim == 3, f'Expect 3D labels (B, T, C), but got {labels.shape}'

    # remove examples which have no splice sites
    has_expr = labels[:, :, 1:].sum((1, 2)) > 0
    logits, labels = logits[has_expr], labels[has_expr]

    B, T, C = logits.shape
    logits, labels = logits.reshape(B * T, C), labels.reshape(B * T, C)

    boundary_mask = labels[:, 1:].sum(1) > 0  # either splice acceptor or donor
    probs = jax.nn.softmax(logits, axis=-1)
    acceptor_probs, donor_probs = probs[boundary_mask, 1], probs[boundary_mask, 2]
    acceptor_labels, donor_labels = labels[boundary_mask, 1], labels[boundary_mask, 2]

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


def top_k_accuracy_per_example(logits, labels, ks=(0.5, 1, 2, 4)):
    """
    Correct version of top-k accuracy, at least relative to intuition/interpretation of the paper.

    In the original source code, top-k accuracy is computed relative to the entire
    batch_size * sequence_length.

    It is probably a more fair evaluation if this is done on a per-example level.
    """
    ...


def fwd_batch(state, inputs):
    """
    Forward a single batch with no gradients
    """

    logits = state.apply_fn(
        {'params': state.params, 'batch_stats': state.batch_stats},
        inputs,
    )

    return logits


def batched_fwd(X, batch_size: int, state: TrainStateWithBN):
    """
    Forward a whole chunk of data (e.g. large segment of a chromosome)
    """
    p_fwd_fn = jax.pmap(fwd_batch, axis_name='batch')

    world_size = jax.device_count()
    assert batch_size % world_size == 0, f"{batch_size=} must be divisible by {world_size=}"
    batch_size_per_device = batch_size // world_size
    shape_prefix = (world_size, batch_size_per_device)
    out = []

    # data parallel full batches
    num_full_batches, num_ragged = divmod(X.shape[0], batch_size)
    for i in range(num_full_batches):
        Xb = X[i * batch_size:(i + 1) * batch_size]
        Xb = Xb.reshape(shape_prefix + Xb.shape[1:])

        out_b = p_fwd_fn(state, Xb).reshape((batch_size, SEQUENCE_LENGTH, -1))
        out.append(out_b)

    if not num_ragged:
        return jnp.concatenate(out)

    # data parallel ragged
    ragged_pmap, ragged_remaining = divmod(num_ragged, world_size)

    if ragged_pmap > 0:
        Xb = (
            X[num_full_batches * batch_size:-ragged_remaining]
            if ragged_remaining > 0 else X[num_full_batches * batch_size:]
        )
        Xb = Xb.reshape((world_size, ragged_pmap) + X.shape[1:])
        out_b = p_fwd_fn(state, Xb).reshape((ragged_pmap * world_size, SEQUENCE_LENGTH, -1))
        out.append(out_b)

    if not ragged_remaining:
        return jnp.concatenate(out)

    # final ragged, just one device
    Xb = X[-ragged_remaining:]
    out_b = fwd_batch(flax.jax_utils.unreplicate(state), Xb)
    out.append(out_b)

    return jnp.concatenate(out)


def eval_dataset(ds: H5SpliceDataset, batch_size: int, state: TrainStateWithBN):
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
        for k, v in batch_results.items():
            if k.endswith('num_true'):
                continue

            splice_type = k.rsplit('_', 2)[0]
            n = batch_results[f'{splice_type}_num_true']
            agg_results[k].update(v, n=n)

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


def test(config):

    assert (
        config.context_length in CONTEXT_LENGTHS
    ), f'{config.context_length=} not permitted, must be one of {CONTEXT_LENGTHS}'
    rng = jax.random.PRNGKey(config.seed)

    world_size = jax.device_count()
    assert (
        config.batch_size % world_size == 0
    ), f"{config.batch_size} must be divisible by {world_size=}"
    batch_size_per_device = config.batch_size // world_size
    test_ds = get_test_dataset(config.eval_path)
    models = [get_conv_model(config.context_length) for _ in range(5)]


def main(argv):
    del argv  # Unused.

    config = FLAGS.config
    np.random.seed(config.seed)
    _ = test(config)


if __name__ == '__main__':
    print(f"Cmd: `python {' '.join(sys.argv)}`")

    FLAGS = flags.FLAGS

    flags.DEFINE_string('workdir', None, 'Directory to store model data.')
    config_flags.DEFINE_config_file(
        'config',
        None,
        'File path to the training hyperparameter configuration.',
        lock_config=True,
    )
    flags.mark_flags_as_required(['config'])

    jax.config.config_with_absl()
    app.run(main)
