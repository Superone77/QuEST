from pathlib import Path
import numpy as np
from typing import Dict
import torch
import torch.distributed as dist

from .shakespeare import get_shakespeare_data
from .wikitext import get_wikitext_data
from .arxiv import get_arxiv_2000, get_arxiv_full
from .openwebtext2 import get_openwebtext2_data
from .redpajama import get_redpajama_data, get_redpajamav2_data
from .slimpajama import get_slimpajama_data
from .c4 import get_c4_data


def get_dataset(args) -> Dict[str, np.ndarray]:
    """Fetch the right dataset given by the args.dataset parameter. The logic for each dataset is
    contained in its own python file. The expected format at the moment is a dictionary of np.memmap
    containing two keys: 'train' and 'val', corresponding to the tokenized training and validation data.
    """
    if args.dataset == "wikitext":
        return get_wikitext_data(args.datasets_dir)
    if args.dataset == "shakespeare-char":
        return get_shakespeare_data(args.datasets_dir)
    if args.dataset == "arxiv2000":
        return get_arxiv_2000(args.datasets_dir)
    if args.dataset == "arxiv":
        return get_arxiv_full(args.datasets_dir)
    if args.dataset == "arxiv+wiki":
        arxiv_data = get_arxiv_full(args.datasets_dir)
        wiki_data = get_wikitext_data(args.datasets_dir)
        train_data = np.concatenate((arxiv_data["train"], wiki_data["train"]))
        val_data = np.concatenate((arxiv_data["val"], wiki_data["val"]))
        return {"train": train_data, "val": val_data}
    if args.dataset == "openwebtext2":
        return get_openwebtext2_data(args.datasets_dir)
    if args.dataset == "redpajama":
        return get_redpajama_data(args.datasets_dir)
    if args.dataset == "redpajamav2":
        return get_redpajamav2_data(args.datasets_dir)
    if args.dataset == "slimpajama":
        return get_slimpajama_data(args.datasets_dir)
    if args.dataset == "c4":
        return get_c4_data(args.datasets_dir, 128)
    else:
        raise NotImplementedError(f"Unknow dataset key '{args.dataset}'")


class DataReader:
    def __init__(
        self,
        data_src,
        batch_size,
        sequence_length,
        seed=1337,
        with_replacement=False,
        auto_shard=True,
        keep_in_ram=False,
    ):
        if isinstance(data_src, (str, Path)):
            self.data_path = Path(data_src)
            self.keep_in_ram = keep_in_ram
            if keep_in_ram:
                self.data = np.array(
                    np.memmap(self.data_path, dtype=np.uint16, mode="r")
                )
            else:
                self.data = None
        elif isinstance(data_src, (np.ndarray, np.memmap)):
            self.data_path = None
            self.data = data_src
            self.keep_in_ram = True

        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.seed = seed
        self.with_replacement = with_replacement

        self.num_tokens = len(self._get_data())

        if auto_shard and dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            print(
                f"Distributed DataReader Initialized for Worker {self.rank}/{self.world_size}"
            )
        else:
            self.world_size = 1
            self.rank = 0

        # Sampling without replacement
        self.last_epoch = None
        self.order = None
        self.epoch_offset = None
        self.step = 0
        self.num_batches_of_seqlen = 0
        if not with_replacement:
            self._shuffle_epoch(0)

    def __len__(self):
        # Length in valid start indices for a sequence
        # Extra -1 to have a valid next token for the final token of the last idx
        return self.num_tokens - self.sequence_length - 1

    def _get_data(self):
        if self.data is not None:
            return self.data
        else:
            # Construct the memmap each time to avoid a memory leak per NanoGPT
            # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
            return np.memmap(self.data_path, dtype=np.uint16, mode="r")

    def __getitem__(self, idx):
        # Return the underlying datapoint, no random sampling, no worker sharding
        assert 0 <= idx < len(self)
        data = self._get_data()
        x = torch.from_numpy(data[idx : idx + self.sequence_length].astype(np.int64))
        y = torch.from_numpy(
            data[idx + 1 : idx + self.sequence_length + 1].astype(torch.int64)
        )
        return x, y

    def set_step(self, step):
        self.step = step

    def sample_batch(self):
        data = self._get_data()

        if self.with_replacement:
            idxs = self._sample_with_replacement(self.step)
        else:
            idxs = self._sample_without_replacement(self.step)
        self.step += 1

        xy = np.stack([data[i : i + self.sequence_length + 1] for i in idxs]).astype(
            np.int64
        )
        x = torch.from_numpy(xy[:, :-1]).contiguous()
        y = torch.from_numpy(xy[:, 1:]).contiguous()
        return x, y

    def _sample_with_replacement(self, idx):
        # Return an array of token indices of length self.batch_size
        # Sampled with replacement, can get repeats at any time
        seed = self.seed + idx * self.world_size + self.rank
        rng = np.random.default_rng(seed)
        return rng.integers(len(self), self.batch_size)

    def _shuffle_epoch(self, epoch):
        seed = self.seed + epoch
        rng = np.random.default_rng(seed)
        # Drop one sequence to allow different offsets per epoch:
        self.order = rng.permutation((len(self)) // self.sequence_length - 1)
        # Shift all sequences in this epoch by this amount:
        self.epoch_offset = rng.integers(self.sequence_length)
        self.last_epoch = epoch
        self.num_batches_of_seqlen = (
            len(self.order) // self.batch_size
        )  # Drops remainder batch

    def _sample_without_replacement(self, step):
        # Return an array of token indices of length self.batch_size
        # Sampled without replacement, cycle all sequences before potential repeats
        # Sequences are randomly offset in every epoch as well
        batch_idx = self.world_size * step + self.rank
        epoch_length = self.num_batches_of_seqlen

        epoch = batch_idx // epoch_length
        if epoch != self.last_epoch:
            self._shuffle_epoch(epoch)
        epoch_idx = batch_idx % epoch_length

        start = epoch_idx * self.batch_size
        end = start + self.batch_size
        return self.order[start:end] * self.sequence_length + self.epoch_offset

    def num_batches(self):
        if self.with_replacement:
            return self.num_tokens // self.batch_size
        return self.num_batches_of_seqlen


class StreamingDataReader:
    def __init__(
        self,
        data_src,
        batch_size,
        sequence_length,
        seed=1337,
        with_replacement=False,
        auto_shard=True,
        buffer_size=1000,  # Number of sequences to keep in memory
    ):
        if isinstance(data_src, (str, Path)):
            self.data_path = Path(data_src)
            self.data = None  # Will be loaded in chunks
        elif isinstance(data_src, (np.ndarray, np.memmap)):
            raise ValueError("StreamingDataReader does not support in-memory data")

        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.seed = seed
        self.with_replacement = with_replacement
        self.buffer_size = buffer_size

        # Get total size of the data
        self.data_mmap = np.memmap(self.data_path, dtype=np.uint16, mode="r")
        self.num_tokens = len(self.data_mmap)
        self.num_sequences = self.num_tokens - self.sequence_length - 1

        if auto_shard and dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            print(f"Distributed StreamingDataReader Initialized for Worker {self.rank}/{self.world_size}")
        else:
            self.world_size = 1
            self.rank = 0

        # Initialize buffer and indices
        self.buffer = []
        self.buffer_indices = []
        self.current_position = 0
        self.rng = np.random.default_rng(seed)
        
        # Fill initial buffer
        self._fill_buffer()

    def _fill_buffer(self):
        """Fill the buffer with new sequences"""
        if self.with_replacement:
            # Sample with replacement
            indices = self.rng.integers(0, self.num_sequences, size=self.buffer_size)
        else:
            # Sample without replacement
            remaining = self.num_sequences - self.current_position
            if remaining < self.buffer_size:
                # Reset position and shuffle
                self.current_position = 0
                indices = np.arange(self.num_sequences)
                self.rng.shuffle(indices)
                indices = indices[:self.buffer_size]
            else:
                indices = np.arange(
                    self.current_position,
                    self.current_position + self.buffer_size
                )
                self.current_position += self.buffer_size

        # Load sequences into buffer
        self.buffer = []
        self.buffer_indices = []
        for idx in indices:
            sequence = self.data_mmap[idx:idx + self.sequence_length + 1]
            self.buffer.append(sequence)
            self.buffer_indices.append(idx)

    def sample_batch(self):
        """Sample a batch of sequences from the buffer"""
        if len(self.buffer) < self.batch_size:
            self._fill_buffer()

        # Sample batch_size sequences from buffer
        batch_indices = self.rng.choice(len(self.buffer), size=self.batch_size, replace=False)
        batch_sequences = [self.buffer[i] for i in batch_indices]
        
        # Remove used sequences from buffer
        for i in sorted(batch_indices, reverse=True):
            del self.buffer[i]
            del self.buffer_indices[i]

        # Convert to tensors
        xy = np.stack(batch_sequences).astype(np.int64)
        x = torch.from_numpy(xy[:, :-1]).contiguous()
        y = torch.from_numpy(xy[:, 1:]).contiguous()
        
        return x, y

    def __len__(self):
        return self.num_sequences

    def num_batches(self):
        return self.num_sequences // self.batch_size
