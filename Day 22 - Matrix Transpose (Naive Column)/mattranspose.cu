#include <cuda.h>
#include <stdio.h>
#define TPB 32
#define NR 4096
#define NC 4096
__global__ void transpose(int *mat, int *out){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < NR && col < NC) {
        out[col * NR + row] = mat[row * NC + col];
    }
}
int main() {
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time_ms = 0.0f;
    int *h_mat, *h_out;
    int *d_mat, *d_out;
    h_mat = (int *)malloc(NR * NC * sizeof(int));
    h_out = (int *)malloc(NC * NR * sizeof(int));
    cudaMalloc((void **)&d_mat, NR * NC * sizeof(int));
    cudaMalloc((void **)&d_out, NC * NR * sizeof(int));
    for (int i = 0; i < NR; i++){
        for (int j = 0; j < NC; j++) {
            h_mat[i * NC + j] = rand() % 10;
        }
    }
    cudaMemcpy(d_mat, h_mat, NR * NC * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, h_out, NC * NR * sizeof(int), cudaMemcpyHostToDevice);
    cudaEventRecord(start, 0);
    dim3 grid((NC + TPB - 1) / TPB, (NR + TPB - 1) / TPB, 1);
    dim3 block(TPB, TPB, 1);
    transpose<<<grid, block>>>(d_mat, d_out);
    cudaEventRecord(stop,  0);
    cudaEventSynchronize(stop);
    cudaMemcpy(h_mat, d_out, NC * NR * sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventElapsedTime(&time_ms, start, stop);
    printf("Time Taken = %0.2f ms\n", time_ms);
    cudaFree(d_mat);
    cudaFree(d_out);
    free(h_mat);
    free(h_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}