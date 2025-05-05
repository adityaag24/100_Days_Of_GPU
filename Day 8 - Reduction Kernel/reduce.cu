#include <stdio.h>
#include <cuda.h>
#define N 8192
#define THREADS_PER_BLOCK 128
__global__ void reduce1(int *g_iData, int *g_result) {
    int bNum = blockIdx.x * blockDim.x;
    int tNum = threadIdx.x;
    int idx  = bNum + tNum;
    if (idx < N){
        atomicAdd(g_result, g_iData[idx]);
    }
}
void runReduction1(int *hData, int h_result) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int *g_iData  = NULL;
    int *g_oData  = NULL;
    int d_result = 0.0;
    int *g_result = NULL;
    float timeTaken = 0.0f;
    cudaMalloc(&g_iData, N * sizeof(int));
    cudaMalloc(&g_oData, N * sizeof(int));
    cudaMalloc(&g_result, sizeof(int));
    cudaMemcpy(g_iData, hData, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaEventRecord(start, 0);
    int blocks    = (N + THREADS_PER_BLOCK - 1)/ THREADS_PER_BLOCK;
    reduce1<<<blocks, THREADS_PER_BLOCK>>>(g_iData, g_result);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeTaken, start, stop);
    cudaMemcpy(&d_result, g_result, sizeof(int), cudaMemcpyDeviceToHost);
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
    runReduction1(hData, hResult);
}