from torch.utils.data import Dataset
import numpy as np

import h5py


class H5SpliceDataset(Dataset):
    """
    Dataset housing examples of nucleotide sequences and their corresponding splice annotations
    (e.g. classification for each nucleotide into `splice_donor`, `splice_acceptor`, `neihter`)
    """
    def __init__(self, h5_path):
        self.h5 = h5py.File(h5_path, 'r')

        # Current dataset uses two separate keys for each chunk
        num_chunks = len(self.h5.keys()) // 2
        chunk_sizes = [self.h5[f'X{i}'].shape[0] for i in range(num_chunks)]
        self.cum_chunk_sizes = np.cumsum(chunk_sizes)

    def __len__(self):
        return self.cum_chunk_sizes[-1]

    def __getitem__(self, idx):
        chunk_idx = np.searchsorted(self.cum_chunk_sizes, idx) - 1
        rel_idx = idx - self.cum_chunk_sizes[chunk_idx]
        X = self.h5[f'X{chunk_idx}'][rel_idx]
        Y = self.h5[f'Y{chunk_idx}'][rel_idx]

        return {'x': X, 'y': Y}
