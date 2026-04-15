#include <stdio.h>
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

__global__ void range_kernel(int *input, int* range_start, int* range_end, int N, int num_bins){
    int global_index = blockDim.x * blockIdx.x + threadIdx.x;
    int local_index = threadIdx.x;
    __shared__ int buffer[2048 + 2]; // 2048 >= threads in a SM
    // you should not init range here becasue you cannot make any assumptions to the execution orders of blocks.
    
    if(global_index < N){
        buffer[local_index + 1] = input[global_index];
    }else{
        buffer[local_index + 1] = -1;
    }
    if(local_index == 0){
        if(global_index == 0) buffer[0] = -1;
        else buffer[0] = input[global_index - 1];
    }
    if(local_index == blockDim.x - 1){
        if(global_index + 1 < N) buffer[blockDim.x + 1] = input[global_index + 1];
        else buffer[blockDim.x + 1] = -1;
    }
    __syncthreads();

    if((buffer[local_index + 1] != buffer[local_index]) && (buffer[local_index + 1] != -1)) range_start[buffer[local_index + 1]] = global_index;
    if((buffer[local_index + 1] != buffer[local_index + 2]) && (buffer[local_index + 1] != -1)) range_end[buffer[local_index + 1]] = global_index;

}

__global__ void histogram_kernel(int* range_start, int* range_end, int* histogram, int num_bins){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < num_bins)
        histogram[idx] = range_end[idx] - range_start[idx] + 1;
}

__global__ void bad_histogram_kernel_1(int *input, int* histogram, int N, int num_bins){
    int global_index = blockDim.x * blockIdx.x + threadIdx.x;
    if(global_index < N) atomicAdd(&histogram[input[global_index]], 1);
}

void bad_histogram_launcher_1(int* input, int* histogram, int N, int num_bins){
    cudaEvent_t start, end;
    float ms;
    cudaMemset(histogram, 0, num_bins);
    cudaEventCreate(&start);cudaEventCreate(&end);
    cudaEventRecord(start);
    bad_histogram_kernel_1<<< (N - 1) / 512 + 1, 512>>>(input, histogram, N, num_bins);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&ms, start, end);
    double data_size_gb = (double)(N + num_bins) * 4 / 268435456;
    double hbm_bw_tb = 3.361; // H800
    // We need to carry 4N bytes from HBM to core and 4num_bins back.
    double ideal_ms = data_size_gb / hbm_bw_tb;
    // bad histogram 1 running time is 34.49801636 ms, input size 0.7451 GB, idealy 0.2217 ms, which reaches 0.6 %
    printf("bad histogram 1 running time is %.8f ms, input size %.4f GB, idealy %.4f ms, which reaches %.1f %\n",ms, data_size_gb, ideal_ms, ideal_ms * 100 / ms);
}

void histogram_launcher(int* input, int* histogram, int N, int num_bins){
    int* range_start,*range_end;
    cudaMalloc(&range_start, sizeof(int) * num_bins);
    cudaMalloc(&range_end, sizeof(int) * num_bins);
    cudaMemset(&range_start, -1, sizeof(int) * num_bins);
    cudaMemset(&range_end, -2, sizeof(int) * num_bins);
    /*
    cudaEvent_t start_1, start_2, end_1, end_2;
    float ms_1, ms_2;
    cudaEventCreate(&start_1);
    cudaEventCreate(&end_1);
    cudaEventRecord(start_1);
    sort_ext(input, N);
    cudaEventRecord(end_1);
    cudaEventSynchronize(end_1);
    cudaEventElapsedTime(&ms_1, start_1, end_1);
    cudaEventCreate(&start_2);
    cudaEventCreate(&end_2);
    cudaEventRecord(start_2);
    */
    range_kernel<<< (N - 1) / 512 + 1, 512>>>(input, range_start, range_end, N, num_bins);
    histogram_kernel<<< (num_bins - 1) / 512 + 1, 512>>>(range_start, range_end, histogram, num_bins);
    /*
    cudaEventRecord(end_2);
    cudaEventSynchronize(end_2);
    cudaEventElapsedTime(&ms_2, start_2, end_2);
    double data_size_gb = (double)(N + num_bins) * 4 / 268435456;
    double hbm_bw_tb = 3.361; // H800
    // We need to carry 4N bytes from HBM to core and 4num_bins back.
    double ideal_ms = data_size_gb / hbm_bw_tb;
    // running time is 0.31065601 ms, input size 0.7451 GB, idealy 0.2217 ms, which reach 71.4 % (not use sort to increase cache hit rate)
    printf("[step1] sort running time is %.8f ms\n",ms_1);
    printf("[step2] histogram running time is %.8f ms, input size %.4f GB, idealy %.4f ms, which reaches %.1f %\n",ms_2, data_size_gb, ideal_ms, ideal_ms * 100 / ms_2);
    */
    cudaFree(range_start);
    cudaFree(range_end);
}


void solve(const int *input, int *histogram, int N, int num_bins)
{
    int *d_input, *d_h;
    cudaSetDevice(0);

    cudaMalloc(&d_input, N * sizeof(int));
    cudaMalloc(&d_h, num_bins * sizeof(int));

    cudaMemcpy(d_input, input, N * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start_1, start_2, end_1, end_2;
    float ms_1, ms_2;
    cudaEventCreate(&start_1);cudaEventCreate(&end_1);cudaEventRecord(start_1);
    thrust::sort(thrust::device, d_input, d_input + N);
    cudaEventRecord(end_1);cudaEventSynchronize(end_1);cudaEventElapsedTime(&ms_1, start_1, end_1);
    // sort();
    cudaEventCreate(&start_2);cudaEventCreate(&end_2);cudaEventRecord(start_2);
    histogram_launcher(d_input, d_h, N, num_bins);
    cudaEventRecord(end_2);cudaEventSynchronize(end_2);cudaEventElapsedTime(&ms_2, start_2, end_2);
    // bad_histogram_launcher_1(d_input, d_h, N, num_bins);
    printf("[step1] sort running time is %.8f ms\n",ms_1);
    printf("[step2] histogram running time is %.8f ms\n",ms_2);

    cudaMemcpy(histogram, d_h, num_bins * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_h);
}

// how to measure the performance of a kernel?
int compare(const void *a, const void *b) {
    return (*(int*)a - *(int*)b);
}

void ast(int* ref, int* result, int N){
    int mismatch = 0;
    for(int i=0;i < N;i++){
        if(ref[i]!=result[i]) mismatch++;
    }
    printf("%d / %d mismatch\n", mismatch, N);
}

int main(){
    int N = 50000000;
    int num_bins = 1024;
    int *arr = (int*)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) {
        arr[i] = rand() % num_bins;
    }
    // qsort(arr, N, sizeof(int), compare);

    int *ref = (int*)malloc(num_bins * sizeof(int));
    int *result_cuda = (int*)malloc(num_bins * sizeof(int));
    memset(ref, 0, sizeof(int) * num_bins);
    for(int i=0; i< N; i++) ref[arr[i]] ++;
    solve(arr, result_cuda, N, num_bins);
    ast(ref, result_cuda, num_bins);
}