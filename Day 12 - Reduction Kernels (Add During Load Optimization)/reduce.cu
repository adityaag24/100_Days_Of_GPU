#include <stdio.h>
#include <cuda.h>
#define THREADS_PER_BLOCK 128
#define N 8192
__global__ void reduce(int *g_iData, int *g_oData) {
    extern __shared__ int sdata[];
    // Load shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    sdata[tid] = (i < N)?g_iData[i] + g_iData[i + blockDim.x] : 0;
    __syncthreads();
 
    // Do reduction in shared memory
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    // Write result for this block to global memory
    if (tid == 0) {
        g_oData[blockIdx.x] = sdata[0];
    }
}
void runReduction(int *hData, int h_result) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int *g_iData = NULL;
    int *g_oData = NULL;
    int d_result = 0;
    float timeTaken = 0.0f;
    cudaMalloc(&g_iData, N * sizeof(int));
    cudaMalloc(&g_oData, N * sizeof(int));
    cudaMemcpy(g_iData, hData, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaEventRecord(start, 0);
    int blocks    = (N + THREADS_PER_BLOCK - 1)/ THREADS_PER_BLOCK / 2;
    int oldBlocks = blocks;
    while (blocks > 0) {
        reduce<<<blocks, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(int)>>>(g_iData,g_oData);
        cudaMemcpy(g_iData, g_oData, N * sizeof(int), cudaMemcpyDeviceToDevice);
        oldBlocks = blocks;
        blocks    = blocks / THREADS_PER_BLOCK / 2;
    }
    if (blocks == 0 && oldBlocks != 1) {
        reduce<<<1, oldBlocks/2, oldBlocks  * sizeof(int)>>>(g_iData, g_oData);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeTaken, start, stop);
    cudaMemcpy(&d_result, &g_oData[0], sizeof(int), cudaMemcpyDeviceToHost);
    if (d_result == h_result) {
        printf("Results Matched.\n");
        printf("Time Taken: %.2f ms\n", timeTaken);
        printf("Bandwidth: %.2f MB/s\n", (N * 2 * sizeof(int)) / (timeTaken * 1e6));
    } else {
        printf("Results Mismatched.\n");
        printf("Expected: %d, Got: %d\n", h_result, d_result);
    }
    cudaFree(g_iData);
    cudaFree(g_oData);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
int main() {
    int hData[N];
    int hResult = 0;
    for (int i = 0; i < N; i++){
        hData[i] = rand() % 10;
        hResult += hData[i];
    }
    runReduction(hData, hResult);
}