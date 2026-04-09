import triton
import triton.language as tl
import torch
import torch.nn.functional as F
import gc
import argparse
import torch.multiprocessing as mp
import json
import os
"""
Special thanks to Alpha8977 for converting the Java RNG program into a C implementation.
The remaining work (implementing the Triton kernel version of the RNG, max index selection)
was done by pseudoFrition.
"""
@triton.jit
def slime_chunk_kernel(
    out_ptr,
    world_seed,
    grid_x: tl.constexpr,
    grid_z: tl.constexpr,
    start_chunkX: tl.constexpr,
    start_chunkZ: tl.constexpr,
):
    # threads
    pid_x = tl.program_id(0)
    pid_z = tl.program_id(1)

    cx = tl.cast(tl.program_id(0) + start_chunkX, tl.int32)
    cz = tl.cast(tl.program_id(1) + start_chunkZ, tl.int32)

    ws = tl.cast(world_seed, tl.uint64)

    # tl.device_print("cx (low)", tl.cast(cx, tl.uint32))
    # tl.device_print("cx (high)", tl.cast(cx >> 32, tl.uint32))
    """
    explicitly use the following cast to align with the Java implementation.
    """
    _term_1 = tl.cast(cx * cx * 4987142, tl.uint64)
    _term_2 = tl.cast(cx * 5947611, tl.uint64)
    _term_3 = tl.cast(cz * cz, tl.uint64) * 4392871
    _term_4 = tl.cast(cz * 389711, tl.uint64)

    seed = (ws + _term_1 + _term_2 + _term_3 + _term_4) ^ 987234911

    # tl.device_print("seed calced (low)", tl.cast(seed, tl.uint32))
    # tl.device_print("seed calced (high)", tl.cast(seed >> 32, tl.uint32))

    A = 25214903917
    C = 11
    M = (1 << 48) - 1

    rng = (seed ^ A) & M # uint64

    flag = True
    val = 0
    """
    in practice, the following code rarely satisfies condition == True. Therefore, we can generally assume 
    that max_itr is aprox 1 - 2, meaning no significant warp divergence will impact performance.
    """
    while flag:
        rng = (rng * A + C) & M # .next update rng
        bits = tl.cast(rng >> 17, tl.int32) # .next return value
        val = bits % 10
        
        condition = (bits - val + 9) < 0

        if (not condition):
            flag = False

    idx = pid_z * grid_x + pid_x
    tl.store(out_ptr + idx, (val == 0).to(tl.int32))

def compute_slime_area(rank:int, out_tensor: torch.Tensor, world_seed: int, patch_size: int, start_chunkX: int, start_chunkZ: int):
    """
    out_tensor must be dtype torch.int32, with shape (patch_size, patch_size), which is used for store the results from kernel.
    The calculation starts from (start_chunkX, start_chunkZ), to (start_chunkX + patch_size, start_chunkZ + patch_size) (exslusive)
    The result is stored in xz manner, i.e., out_tensor[row, col] is from chunkX = col + start_chunkX, chunkZ = row + start_chunkZ
    -------(0,n)-----------> +x
    |        |
    |        |
  (m,0)----(m, n) (row-col indexing: (m,n), xz indexing: (n,m))
    |
    |
    |
    +z
    """
    assert out_tensor.shape == (patch_size, patch_size,)
    assert out_tensor.dtype == torch.int32
    # launching kernel!
    slime_chunk_kernel[(patch_size, patch_size)](
        out_tensor,
        world_seed,
        patch_size, patch_size,
        start_chunkX, start_chunkZ,
        # num_warps=1,
        # num_stages=1,
    )
    return 

# @torch.compile(fullgraph=True, mode="max-autotune", dynamic=True)
def selection(out_tensor: torch.Tensor, half_gs:int, gs: int):
    prefix_S = F.pad(out_tensor, (half_gs + 1, half_gs, half_gs + 1, half_gs), mode='constant', value=0).cumsum_(dim=0).cumsum_(dim=1)
    acc_grid = prefix_S[gs:,  gs:]  -  prefix_S[:-gs, gs:]  -  prefix_S[gs:,  :-gs] + prefix_S[:-gs, :-gs]
    row, col = torch.unravel_index(torch.argmax(acc_grid), acc_grid.shape)
    return row, col, acc_grid[row,col]

def task(rank: int, seed:int, chunkXs: list[int], chunkZs: list[int], half_gs: int = 8, patch_size: int = 32767,  num_processes: int=8):
    # from tqdm import tqdm
    cur_x = None 
    cur_z = None 
    cur_max = -1
    device = f"cuda:{rank}"
    torch.cuda.set_device(device)
    # reuseable tensor!
    out_tensor = torch.empty(size=(patch_size,patch_size,), dtype=torch.int32, device=device)
    gs = half_gs * 2 + 1
    
    cur_x = None
    cur_z = None 
    cur_max = -1


    for idx, (chunkX, chunkZ) in enumerate(zip(chunkXs, chunkZs)):
        if (idx % num_processes != rank):
            continue

        compute_slime_area(rank, out_tensor, seed, patch_size, chunkX, chunkZ)
        # directly read result from out_tensor is okay! Be careful for OOM!
        # print(out_tensor)

        # TODO torch.complie following code sections
        """
        prefix_S = F.pad(out_tensor, (half_gs + 1, half_gs, half_gs + 1, half_gs), mode='constant', value=0).cumsum_(dim=0).cumsum_(dim=1)
        acc_grid = prefix_S[gs:,  gs:]  -  prefix_S[:-gs, gs:]  -  prefix_S[gs:,  :-gs] + prefix_S[:-gs, :-gs]
        row, col = torch.unravel_index(torch.argmax(acc_grid), acc_grid.shape)
        val = acc_grid[row,col].clone()
        """
        
        row, col, val = selection(out_tensor, half_gs, gs)
        row = row.item()
        col = col.item()
        val = val.item()

        # forcing gc
        # del prefix_S, acc_grid
        # gc.collect()
        # torch.cuda.empty_cache()

        if val > cur_max:
            cur_x = col + chunkX
            cur_z = row + chunkZ
            cur_max = val
    
    # print("putting result json file")
    # We only want to find 5*5 full of slime chunks
    if val < (gs * gs):
        return
    
    with open(os.path.join(".", f"_{rank}_result.json"), "w") as f:
        json.dump((cur_x, cur_z, cur_max), f)
        print(f"cuda:{rank}: (chunkX, chunkZ, count) = {(cur_x, cur_z, cur_max)}")

# Example execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_proc', type=int, default= 8)
    parser.add_argument('--search_range_x_start', type=int, default= -1875000)
    parser.add_argument('--search_range_x_end', type=int, default= 1875000)
    parser.add_argument('--search_range_z_start', type=int, default= -1875000)
    parser.add_argument('--search_range_z_end', type=int, default= 1875000)
    parser.add_argument('--seed', type=int, default = 0)
    parser.add_argument('--patch_size', type=int, default= 32767)
    parser.add_argument('--gs', type=int, default= 8)
    
    args = parser.parse_args()

    search_range_x_start = args.search_range_x_start
    search_range_x_end = args.search_range_x_end 
    search_range_z_start = args.search_range_z_start
    search_range_z_end = args.search_range_z_end
    patch_size = args.patch_size

    all_chunkx = []
    all_chunkz = []

    x_starts = range(search_range_x_start, search_range_x_end, patch_size)
    z_starts = range(search_range_x_start, search_range_x_end, patch_size)

    for x in x_starts:
        for z in z_starts:
            all_chunkx.append(x)
            all_chunkz.append(z)
    
    print(f"There are total {len(all_chunkx)} patches, each patch is of size {patch_size}x{patch_size} chunks.")
    num_processes = args.num_proc

    max_itr = 512

    for epoch in range(max_itr):
        print(f"epoch {epoch}: launching {num_processes} cuda processes with seed {args.seed + epoch}")
        # To be run on 8 GPUs, esitimated memory per GPU: >~ 17 GB
        # def task(rank: int, result_queue: mp.Queue, seed:int, chunkXs: list[int], chunkZs: list[int], half_gs: int = 8, patch_size: int = 32767):
        mp.spawn(fn=task, args=(args.seed + epoch, all_chunkx, all_chunkz, args.gs, patch_size, num_processes), nprocs=num_processes, join=True)

    print("Work done.")