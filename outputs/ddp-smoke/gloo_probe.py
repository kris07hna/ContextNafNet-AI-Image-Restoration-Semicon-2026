import os
import torch
import torch.distributed as dist

dist.init_process_group(backend="gloo", world_size=int(os.environ["WORLD_SIZE"]), rank=int(os.environ["RANK"]))
rank = dist.get_rank()
cpu = torch.ones(4) * (rank + 1)
dist.all_reduce(cpu)
print(f"rank={rank} cpu_allreduce_ok={cpu.tolist()}", flush=True)
gpu = torch.ones(4, device="cuda") * (rank + 1)
dist.all_reduce(gpu)
print(f"rank={rank} cuda_allreduce_ok={gpu.tolist()}", flush=True)
dist.destroy_process_group()
