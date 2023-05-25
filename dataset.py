from typing import Union

import h5py
import numpy as np
from torch.utils.data import Dataset


class H5SpliceDataset(Dataset):
    """
    Dataset housing examples of nucleotide sequences and their corresponding splice annotations
    (e.g. classification for each nucleotide into `splice_donor`, `splice_acceptor`, `neither`)

    Uses a pointer to the h5 file and a subset of indices to include.
    """
    def __init__(self, h5_path, indices: Union[list, np.ndarray]):
        self.h5 = h5py.File(h5_path, 'r')
        self.indices = indices

        chunk_sizes = [self.h5[f'X{i}'].shape[0] for i in range(indices)]
        self.cum_chunk_sizes = np.cumsum(chunk_sizes)

        print(
            f"Initiated {self.__class__} using {len(indices)} chunks from {h5_path} "
            f"({len(self)} examples)"
        )

    def __len__(self):
        return self.cum_chunk_sizes[-1]

    def __getitem__(self, idx):
        # Relative indices: index of chunk to sample, index within chunk
        chunk_idx = np.searchsorted(self.cum_chunk_sizes, idx) - 1
        rel_idx = idx - self.cum_chunk_sizes[chunk_idx]

        # Absolute index: which of global chunks we sampled
        abs_idx = self.indices[chunk_idx]

        X = self.h5[f'X{abs_idx}'][rel_idx]
        Y = self.h5[f'Y{abs_idx}'][rel_idx]

        return {'x': X, 'y': Y}


def get_train_val_datasets(h5_path, val_proportion=0.9):
    with h5py.File(h5_path, 'r') as f:
        num_chunks = len(f.keys()) // 2

    all_idx = list(range(num_chunks))
    np.random.shuffle(all_idx)
    tr_idx = all_idx[:int(val_proportion * num_chunks)]
    val_idx = all_idx[int(val_proportion * num_chunks):]

    return H5SpliceDataset(h5_path, tr_idx), H5SpliceDataset(h5_path, val_idx)
