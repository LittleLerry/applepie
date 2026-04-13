#include <cuda_runtime.h>
#include <stdio.h>

__global__ void gemm_kernel(const float *A, const float *B, float *result, int M, int N, int K)
{
    // (M, K) @ (K, N)
    // fixed 16 * 16 shared memory
    // This is my baseline programme
    const int PATCH_SIZE = 16;
    __shared__ float buffer_A[PATCH_SIZE][PATCH_SIZE]; // it is shared IF __extern__
    __shared__ float buffer_B[PATCH_SIZE][PATCH_SIZE];

    int num_blocks_row = (M - 1) / PATCH_SIZE + 1;
    int num_blocks_col = (N - 1) / PATCH_SIZE + 1;

    int row_start_index = blockIdx.x / num_blocks_col * PATCH_SIZE;
    int col_start_index = blockIdx.x % num_blocks_col * PATCH_SIZE;

    int inner_row_start_index = threadIdx.x / PATCH_SIZE;
    int inner_col_start_index = threadIdx.x % PATCH_SIZE;

    float _temp = 0.0f;
    int row_, col_;
    for (int i = 0; i < num_blocks_col; i++)
    {
        // read buffers
        row_ = row_start_index + inner_row_start_index;
        col_ = inner_col_start_index + i * PATCH_SIZE;
        if ((row_ < M) && (col_ < K))
            buffer_A[inner_row_start_index][inner_col_start_index] = A[row_ * K + col_];
        else
            buffer_A[inner_row_start_index][inner_col_start_index] = 0.0f;

        row_ = inner_row_start_index + i * PATCH_SIZE;
        col_ = col_start_index + inner_col_start_index;
        if ((row_ < K) && (col_ < N))
            buffer_B[inner_row_start_index][inner_col_start_index] = B[row_ * N + col_];
        else
            buffer_B[inner_row_start_index][inner_col_start_index] = 0.0f;

        // buffer_A[inner_row_start_index][inner_col_start_index] = A[(row_start_index + inner_row_start_index) * N + inner_col_start_index + i * PATCH_SIZE];
        // buffer_B[inner_row_start_index][inner_col_start_index] = B[(inner_row_start_index + i * PATCH_SIZE) * N + col_start_index + inner_col_start_index];
        __syncthreads();

        for (int k = 0; k < PATCH_SIZE; k++)
            _temp += buffer_A[inner_row_start_index][k] * buffer_B[k][inner_col_start_index];
        __syncthreads();
    }
    row_ = row_start_index + inner_row_start_index;
    col_ = col_start_index + inner_col_start_index;
    if ((row_ < M) && (col_ < N))
        result[row_ * N + col_] = _temp;
    // result[(row_start_index + inner_row_start_index) * N + col_start_index + inner_col_start_index] = _temp;
}

int main()
{
    const int M = 290;
    const int N = 153;
    const int K = 128;

    float *A = (float *)malloc(M * K * sizeof(float));
    float *B = (float *)malloc(K * N * sizeof(float));
    float *R = (float *)malloc(M * N * sizeof(float));

    float *GT = (float *)malloc(M * N * sizeof(float));

    for (int i = 0; i < M; i++)
        for (int j = 0; j < K; j++)
            A[i * K + j] = ((float)i + j) / 10;

    for (int i = 0; i < K; i++)
        for (int j = 0; j < N; j++)
            A[i * N + j] = ((float)i - j) / 10;

    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            GT[i * N + j] = 0.0f;

    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++){
            for (int k = 0; k < K; k++)
                GT[i * N + j] += A[i * K + k] * B[k * N + j];
            R[i * N + j] = 1.0f; // trash data
        }

    float *d_A, *d_B, *d_R;

    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_R, M * N * sizeof(float));

    cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    int PATCH_SIZE = 16;
    int block_shape = PATCH_SIZE * PATCH_SIZE; // 0 \leq threadIdx < 256
    int grid_shape = ((M - 1) / PATCH_SIZE + 1) * ((N - 1) / PATCH_SIZE + 1);
    // ===========================
    float error = 0.0f;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            error += (GT[i * N + j] - R[i * N + j]) * (GT[i * N + j] - R[i * N + j]);
        }
    }
    printf("before abs error = %.8f\n", error);
    // ===========================
    printf("launching kernels with (%d, %d)\n", grid_shape, block_shape);
    gemm_kernel<<<grid_shape, block_shape>>>(d_A, d_B, d_R, M, N, K);
    cudaMemcpy(R, d_R, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    // ===========================
    error = 0.0f;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            error += (GT[i * N + j] - R[i * N + j]) * (GT[i * N + j] - R[i * N + j]);
        }
    }
    printf("after abs error = %.8f\n", error);

    free(A);
    free(B);
    free(R);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_R);
}