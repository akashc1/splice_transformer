from collections import defaultdict
import json
import logging

import chex
import colorama
import jax
from jax import numpy as jnp
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

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
    Forward a single batch
    """

    logits = state.apply_fn(
        {'params': state.params, 'batch_stats': state.batch_stats},
        inputs,
    )

    return logits


def eval_dataset(dl: DataLoader, state: TrainStateWithBN):
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
    assert dl.batch_size % world_size == 0, f"{dl.batch_size=} must be divisible by {world_size=}"
    batch_size_per_device = dl.batch_size // world_size
    shape_prefix = (world_size, batch_size_per_device)

    # NOTE: cannot pmap accuracy function because it cannot trivially be jit'd.
    # However we do still make the forward pass data-parallel
    p_fwd_fn = jax.pmap(fwd_batch, axis_name='batch')

    agg_results = defaultdict(AverageMeter)
    for batch in tqdm(dl, desc='Evaluating dataset'):
        inp, label = batch['x'], batch['y']
        inp = inp.reshape(shape_prefix + inp.shape[1:])

        logits = p_fwd_fn(state, inp).reshape(label.shape)
        batch_results = top_k_accuracy(logits, label)
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
