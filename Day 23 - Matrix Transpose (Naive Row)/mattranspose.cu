#include <cuda.h>
#include <stdio.h>
#define TPB 32
#define NR 4096
#define NC 4096
__global__ void transposeNativeRow(int *mat, int *out){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < NR && col < NC) {
        out[col * NR + row] = mat[row * NC + col];
    }
}
__global__ void transposeNativeCol(int *mat, int *out){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < NR && col < NC) {
        out[row * NC + col] = mat[col * NR + row];
    }
}
int main() {
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time_NativeRow = 0.0f;
    float time_NativeCol = 0.0f;
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
    transposeNativeRow<<<grid, block>>>(d_mat, d_out);
    cudaEventRecord(stop,  0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_NativeRow, start, stop);
    cudaMemcpy(h_mat, d_out, NC * NR * sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventRecord(start, 0);
    transposeNativeCol<<<grid, block>>>(d_mat, d_out);
    cudaEventRecord(stop,  0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_NativeCol, start, stop);
    printf("Time Taken by Native Row Kernel (Load by Row, Store by Col) = %0.2f ms\n", time_NativeRow);
    printf("Time Taken by Native Col Kernel (Load by Col, Store by Row) = %0.2f ms\n", time_NativeCol);
    cudaFree(d_mat);
    cudaFree(d_out);
    free(h_mat);
    free(h_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}