import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import itertools
import time

def setup(rank, world_size, backend):
    # master's ip and port
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    # Each of these worker processes belong to a process group, which is initialized via dist.init_process_group.
    # blocked to init process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def distributed_demo(rank, world_size, sq_data_size = 1024, backend = "gloo", warmup = 5): # rank is extra args passed by torch.multiprocessing.spawn
    setup(rank, world_size, backend)
    device = f"cuda:{rank}" if (backend == "nccl") else "cpu"
    torch.set_default_device(device)
    data = torch.randn(size=(sq_data_size, sq_data_size), dtype=torch.float32, device=device)

    # warm up
    for _ in range(warmup):
        dist.all_reduce(data, async_op=False)
        if backend == "nccl":
            torch.cuda.synchronize()
    
    # record time
    start = time.time()
    dist.all_reduce(data, async_op=False)
    if backend == "nccl":
        torch.cuda.synchronize() # called every process?
    end = time.time()
    
    # log
    duration = torch.tensor(end - start)
    durations = [torch.zeros_like(duration) for _ in range(world_size)]

    # must be called for all !
    # dist.all_gather_object(durations, duration) # dist.all_gather_object(durations, duration) blocked forever in nccl backend....?
    dist.all_gather(durations, duration)
    # no need for sync????????????????? not sure
    if rank == 0:
        print(f"[rank{rank}]: Avg time {torch.tensor(durations).mean()} sec. Datasize {sq_data_size ** 2 * 4 // (1024 ** 2)}MB. Backend {backend}. World_size {world_size}")

    # See https://pytorch.org/docs/stable/distributed.html#shutdown 
    # The call should be made once per trainer-process, not at the outer process-launcher level.
    dist.barrier()
    torch.distributed.destroy_process_group()


# 1) Whenever possible, run benchmarks on the same machine to facilitate controlled comparisons.
# 2) Perform several warm-up steps before timing the operation of interest. This is especially important for
#    NCCL communication calls. 5 iterations of warmup is generally sufficient.
# 3) Call torch.cuda.synchronize() to wait for CUDA operations to complete when benchmarking on
#    GPUs. Note that this is necessary even when calling communication operations with async_op=False,
#    which returns when the operation is queued on the GPU (as opposed to when the communication
#    actually finishes).
# 4) Timings may vary slightly across different ranks, so it’s common to aggregate measurements across
#    ranks to improve estimates. You may find the all-gather collective (specifically the dist.all_gather_object 
#    function) to be useful for collecting results from all ranks.
# 5) Ingeneral, debug locally with Gloo on CPU, and then as required in a given problem, benchmark with
#    NCCL on GPU. Switching between the backends just involves changing the init_process_group call
#    and tensor device casts.



# See https://docs.pytorch.org/tutorials/intermediate/dist_tuto.html#initializing-processing-group 
if __name__ == "__main__":
    # world_size = 4
    warmup = 5
    world_sizes = [2, 4, 6]
    backends = ["nccl", "gloo"]
    sq_data_sizes = [512, 512 * 3, 5120, 5120 * 3]
    # launch extra 4 procs (main proc not included), 1 of them is master(rank 0) and 3 are workers
    for backend, world_size in itertools.product(backends, world_sizes):
        for ds in sq_data_sizes:
            # mp.spawn spawns some processes
            mp.spawn(fn=distributed_demo, args=(world_size, ds, backend, warmup), nprocs=world_size, join=True)
            # all processes have been returned and exited, processes' destorying happens immediately when they return.

            # it may conatins bad locks, uses netstat -tulnp | grep 29500, then kill -9 pid.


