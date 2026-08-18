from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.utils.data import Sampler


@dataclass(frozen=True)
class DistributedInfo:
    enabled: bool = False
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    cuda: bool = False

    @property
    def primary(self) -> bool:
        return self.rank == 0

    @property
    def device(self) -> torch.device:
        if not self.cuda:
            return torch.device("cpu")
        # Wrapping by the visible device count lets more ranks than GPUs share the hardware, which is
        # what makes a two-process smoke test possible on a single card; with one rank per GPU, as on
        # the two-GPU target, the wrap is the identity.
        return torch.device(f"cuda:{self.local_rank % torch.cuda.device_count()}")


def initialize(requested: str = "auto") -> DistributedInfo:
    # torchrun exports these variables; their absence means an ordinary single-process run. Detecting
    # the mode from the environment keeps one training script and one config valid for both, so a
    # single-GPU debug run and a two-GPU Kaggle run differ only in how they are launched.
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    cuda = requested != "cpu" and torch.cuda.is_available()
    if world_size <= 1:
        return DistributedInfo(cuda=cuda)
    if cuda:
        torch.cuda.set_device(local_rank % torch.cuda.device_count())
    if not dist.is_initialized():
        # nccl is the right backend wherever it exists, but it is Linux only, so a CUDA run on Windows
        # has to fall back to gloo. gloo reduces CUDA tensors too, just without the peer-to-peer path.
        backend = "nccl" if cuda and dist.is_nccl_available() else "gloo"
        dist.init_process_group(backend=backend, world_size=world_size, rank=rank)
    return DistributedInfo(True, rank, local_rank, world_size, cuda)


def shutdown() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def reduce_sum(values: dict[str, float], info: DistributedInfo) -> dict[str, float]:
    if not info.enabled:
        return dict(values)
    keys = sorted(values)
    tensor = torch.tensor([values[key] for key in keys], dtype=torch.float64, device=info.device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return dict(zip(keys, tensor.tolist()))


class ShardSampler(Sampler[int]):
    # DistributedSampler pads the final shard by repeating samples so every rank sees the same count,
    # which biases any average computed over the shards. Interleaving without padding hands each rank a
    # disjoint slice instead, so summing per-rank totals and counts reproduces the single-process
    # validation metric exactly rather than approximately.
    def __init__(self, count: int, info: DistributedInfo):
        self.indices = list(range(info.rank, count, info.world_size))

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)
