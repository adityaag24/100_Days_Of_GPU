#include <stdio.h>
#include <cuda.h>
#define N   (1e8)
#define ALPHA 0.01f
#define TPB 32
// Error checking macro
#define CHECK_CUDA_ERROR(call) \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s, line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}
__host__ void leakyReLuCPU(float *input) {
    for (int i = 0; i < N; i++){
        input[i] = (input[i] > 0)? input[i]: input[i] * ALPHA;
    }
}
__global__ void leakyReLuGPU(float *input) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        input[tid] = (input[tid] > 0)? input[tid]: input[tid] * ALPHA;
    }
}
int main() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float *h_copy  = (float *)malloc(N * sizeof(float));
    float *h_input = (float *)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++){
        h_input[i] = (float)(rand()) / (float)(RAND_MAX);
    }
    clock_t hostT;
    hostT = clock();
    leakyReLuCPU(h_copy);
    hostT = clock() - hostT;
    double hostTime = ( ((double)hostT)/CLOCKS_PER_SEC ) * 1e3;
    float *d_input = NULL;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_input, N * sizeof(float)));
    dim3 grid ((N + TPB - 1)/TPB, 1, 1);
    dim3 block(TPB, 1, 1);
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));
    leakyReLuGPU<<<grid, block>>>(d_input);
    CHECK_CUDA_ERROR(cudaMemcpy(h_input, d_input, N * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    float timeTaken;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&timeTaken, start, stop));
    int nElements = N;
    printf("Time Taken for %d elements on CPU = %.2f ms\n", nElements, hostTime);
    printf("Time Taken for %d elements on GPU = %.2f ms\n", nElements, timeTaken);
    printf("Bandwidth on CPU = %.2f MB/s\n", (2 * N * sizeof(float)) / (hostTime  * (1<<20)));
    printf("Bandwidth on GPU = %.2f MB/s\n", (2 * N * sizeof(float)) / (timeTaken * (1<<20)));
    printf("Speedup Obtained = %d\n", ((int)(hostTime / timeTaken)));
    free(h_input);
    cudaFree(d_input);
}