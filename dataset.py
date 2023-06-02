from typing import Union

import h5py
import numpy as np
from torch.utils.data import Dataset

from constants import MAX_CONTEXT_LENGTH


def clip_data(x, y, context_length):
    clip = (MAX_CONTEXT_LENGTH - context_length) // 2
    if clip != 0:
        x = x[..., clip:-clip, :]
    return x, y


class H5SpliceDataset(Dataset):
    """
    Dataset housing examples of nucleotide sequences and their corresponding splice annotations
    (e.g. classification for each nucleotide into `splice_donor`, `splice_acceptor`, `neither`)

    Uses a pointer to the h5 file and a subset of indices to include.
    """
    def __init__(
        self,
        h5_path,
        indices: Union[list, np.ndarray],
        context_length: int,
        load_chunks: bool = False,
    ):
        self.h5 = h5py.File(h5_path, 'r')
        self.indices = indices
        self.context_length = context_length
        self.load_chunks = load_chunks

        chunk_sizes = [self.h5[f'X{i}'].shape[0] for i in indices]
        self.cum_chunk_sizes = np.cumsum(chunk_sizes)

        print(
            f"Initiated {self.__class__} using {len(indices)} chunks from {h5_path} "
            f"({len(self)} examples)"
        )

    def __len__(self):
        return len(self.indices) if self.load_chunks else self.cum_chunk_sizes[-1]

    def __getitem__(self, idx):
        # Return example of whole chunk
        if self.load_chunks:
            chunk_idx = self.indices[idx]
            X = self.h5[f'X{chunk_idx}']
            Y = self.h5[f'Y{chunk_idx}'][0]
            X, Y = clip_data(X, Y, self.context_length)
            return {'x': X, 'y': Y}

        # Relative indices: index of chunk to sample, index within chunk
        chunk_idx = np.searchsorted(self.cum_chunk_sizes, idx, side='right')

        if chunk_idx == 0:
            rel_idx = idx
            abs_idx = self.indices[0]
        else:
            rel_idx = idx - self.cum_chunk_sizes[chunk_idx - 1]
            abs_idx = self.indices[chunk_idx]

        X = self.h5[f'X{abs_idx}'][rel_idx]
        Y = self.h5[f'Y{abs_idx}'][0, rel_idx]  # Dataset generation code does this, idk why
        X, Y = clip_data(X, Y, self.context_length)

        return {'x': X, 'y': Y}


def get_train_val_datasets(h5_path, context_length, val_proportion=0.9):
    with h5py.File(h5_path, 'r') as f:
        num_chunks = len(f.keys()) // 2

    all_idx = list(range(num_chunks))
    np.random.shuffle(all_idx)
    tr_idx = all_idx[:int(val_proportion * num_chunks)]
    val_idx = all_idx[int(val_proportion * num_chunks):]

    return (
        H5SpliceDataset(h5_path, tr_idx, context_length),
        H5SpliceDataset(h5_path, tr_idx, context_length, load_chunks=True),
        H5SpliceDataset(h5_path, val_idx, context_length, load_chunks=True),
    )


def get_test_dataset(h5_path, context_length):

    with h5py.File(h5_path, 'r') as f:
        num_chunks = len(f.keys()) // 2

    idx = list(range(num_chunks))
    return H5SpliceDataset(h5_path, idx, context_length, load_chunks=True)
