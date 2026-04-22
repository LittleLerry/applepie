#include <stdint.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <climits>
#include <iostream>
#include <fstream>
#include <vector>
#include <unistd.h>
#include <sys/wait.h>
#include <cerrno>
#include <cstring>


#define patch_size 64  // can change
#define true_patch_size 60  // can change
#define block_size 128 // cannot change

__device__ __forceinline__ bool is_slime_chunk(int32_t chunkX, int32_t chunkZ, uint64_t seed)
{

    uint64_t A = ((uint64_t)25214903917);
    uint64_t C = 11;
    uint64_t M = (((uint64_t)1 << 48) - 1);

    uint64_t init_seed = seed +
                             (uint64_t)(chunkX * chunkX * 4987142) +
                             (uint64_t)(chunkX * 5947611) +
                             (uint64_t)(chunkZ * chunkZ) * 4392871 +
                             (uint64_t)(chunkZ * 389711) ^
                         987234911;

    uint64_t rng = (init_seed ^ A) & M;

    int32_t bits, val;
    do
    {
        rng = (rng * A + C) & M;
        bits = rng >> 17;
        val = bits % 10;
    } while (bits - val + 9 < 0);

    return val == 0;
}

__device__ __forceinline__ uint32_t get_lane_mask(uint32_t n)
{
    uint32_t start = (n >= 2) * (n - 2);
    uint32_t end = (n <= 29) * (n + 2) + (n > 29) * 31;
    return (((uint32_t)1 << (end - start + 1)) - (uint32_t)1) << start;
}

__global__ void slime_kernel(int32_t offset_chunkX, int32_t offset_chunkZ, uint64_t seed, void *out)
{
    __shared__ uint8_t max_block[4];

    uint32_t lane = threadIdx.x % 32;
    uint32_t warp_id = threadIdx.x / 32;

    uint32_t col = blockIdx.x * true_patch_size + offset_chunkX + (warp_id % 2) * 32 - 2;
    uint32_t row = blockIdx.y * true_patch_size + offset_chunkZ + lane + (warp_id / 2) * 32 - 2;

    uint32_t slime_distribution = 0;
#pragma unroll
    for (int i = 0; i < 32; i++)
    {
        slime_distribution = (slime_distribution << 1) + is_slime_chunk(col + i, row, seed);
    }
    __syncwarp();

    uint32_t lane_mask = get_lane_mask(lane);
    uint32_t slice_mask = 0x1;

    uint8_t stack[5] = {0};
    uint8_t sp = 0;
    uint8_t max_warp = 0;

#pragma unroll
    for (int i = 0; i < 32; i++)
    {
        uint32_t slice = __ballot_sync(0xffffffff, (slime_distribution & slice_mask));
        stack[sp] = __builtin_popcount(slice & lane_mask);
        sp = (sp + 1) % 5;
        max_warp = max(stack[0] + stack[1] + stack[2] + stack[3] + stack[4], max_warp);
        slice_mask = (slice_mask << 1);
    }

#pragma unroll
    for (int i = 16; i >= 1; i /= 2)
    {
        max_warp = max(max_warp, __shfl_down_sync(0xffffffff, max_warp, i));
    }

    if (lane == 0)
        max_block[warp_id] = max_warp;
    __syncthreads();
    if (threadIdx.x == 0)
    // avoid atomic add!
    // overlap gloal write with the computation
        *((int *)out + blockIdx.x * gridDim.x + blockIdx.y) = max(max(max_block[0], max_block[1]), max(max_block[2], max_block[3]));
}

void task(int rank, uint64_t seed_start, uint64_t seed_end){
    cudaSetDevice(rank);
    uint32_t grid_dim = 62500;
    dim3 gridDim(grid_dim, grid_dim);
    int *d_o;
    cudaMalloc(&d_o, sizeof(int) * grid_dim * grid_dim);

    int max_val = -1;
    uint64_t seed;
    std::ofstream file;
    file.open(std::to_string(rank) + ".txt", std::ios::out);

    for(uint64_t i = seed_start; i < seed_end; i++){
        slime_kernel<<<gridDim, block_size>>>(-1875000, -1875000, i, d_o);
        int temp = thrust::reduce(thrust::device, d_o, d_o + grid_dim * grid_dim, std::numeric_limits<int>::min(), thrust::maximum<int>());
        if(temp > max_val){
            max_val = temp;
            seed = i;
            file << "New max found for seed " + std::to_string(i) + ", max = " + std::to_string(max_val) << std::endl;
            file.flush();
        }
    }
    cudaFree(d_o);
    file.close();
}

int main() {
    int NUM_WORKERS = 8;
    int WORKLOADES = 20000; // ~1 week
    std::vector<pid_t> child_pids;
    child_pids.reserve(NUM_WORKERS);

    for (int i = 0; i < NUM_WORKERS; ++i) {
        pid_t pid = fork();
        if (pid < 0) {
            return EXIT_FAILURE;
        } else if (pid == 0) {
            task(i, i * WORKLOADES, (i+1) * WORKLOADES);
            _exit(EXIT_SUCCESS);
        } else {
            child_pids.push_back(pid);
        }
    }

    for (pid_t pid : child_pids) {
        int status = 0;
        waitpid(pid, &status, 0);

        if (WIFEXITED(status)) {
            std::cerr << "[Main] pid " << pid << " exited (code=" << WEXITSTATUS(status) << ")\n";
        } else if (WIFSIGNALED(status)) {
            std::cerr << "[Main] pid " << pid << " terminated (sig=" << WTERMSIG(status) << ")\n";
        } else {
            std::cerr << "[Main] pid " << pid << " aborted\n";
        }
    }
    std::cerr << "[Main] done\n";
}