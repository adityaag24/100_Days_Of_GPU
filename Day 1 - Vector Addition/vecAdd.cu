#include <cuda.h>
#include <stdio.h>
#define N   1024
#define TPB 64
#define BLOCKS (N + TPB - 1) / TPB
__global__ void vecAdd(int *a, int *b, int *c){
    int bNum = blockIdx.z  * (gridDim.y  * gridDim.x)  + blockIdx.y  * gridDim.x  + blockIdx.x;
    int tNum = threadIdx.z * (blockDim.y * blockDim.x) + threadIdx.y * blockDim.x + threadIdx.x;
    int idx  = bNum * (blockDim.z * blockDim.y * blockDim.x) + tNum;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}
int main() {
    int *h_arr = (int *)malloc(N * sizeof(int));
    int *d_arr;
    cudaError_t err;
    err = cudaMalloc((void **)&d_arr, N * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error allocating device memory: %s\n", cudaGetErrorString(err));
        return -1;
    }
    for (int i = 0; i < N; i++) {
        h_arr[i] = i;
    }
    err = cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error copying data to device: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        return -1;
    }
    int *d_out;
    err = cudaMalloc((void **)&d_out, N * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error allocating output device memory: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        return -1;
    }
    vecAdd<<<BLOCKS, TPB>>>(d_arr, d_arr, d_out);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error launching kernel: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        cudaFree(d_out);
        return -1;
    }
    err = cudaMemcpy(h_arr, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error copying data to host: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        cudaFree(d_out);
        return -1;
    }
    for (int i = 0; i < N; i++) {
        if (h_arr[i] != 2 * i) {
            fprintf(stderr, "Error: h_arr[%d] = %d, expected %d\n", i, h_arr[i], 2 * i);
            free(h_arr);
            cudaFree(d_arr);
            cudaFree(d_out);
            return -1;
        }
    }
    printf("All values are correct!\n");
}