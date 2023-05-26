import jax
from sklearn.metrics import average_precision_score
from jax import numpy as jnp
import chex


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

        sorted_p_idx = jnp.argsort(p)

        for k in ks:
            top_p_idx = sorted_p_idx[-int(k * n_true):]
            den = min(n_true, len(top_p_idx))
            acc = jnp.size(jnp.intersect1d(top_p_idx, true_idx)) / den
            thresh = p[sorted_p_idx[-int(k * n_true)]]

            results.update({
                f'{name}_{k}_accuracy': acc,
                f'{name}_{k}_threshold': float(thresh),
                f'{name}_num_true': n_true,
            })

        results['f{name}_auc'] = average_precision_score(l, p)

    return results
