#include <stdio.h>
#include <cuda.h>
__global__ void encryptKernel(char *message, int length) {
    int bNum = blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y * gridDim.x + blockIdx.x;
    int tNum = threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
    int tid  = bNum * (blockDim.x * blockDim.y * blockDim.z) + tNum;
    if (tid < length){
        if (message[tid] == 'z'){
            message[tid] = 'a';
        } else {
            message[tid] = message[tid] + 1;
        }
    }
}
__global__ void decryptKernel(char *message, int length) {
    int bNum = blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y * gridDim.x + blockIdx.x;
    int tNum = threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
    int tid = bNum * (blockDim.x * blockDim.y * blockDim.z) + tNum;
    if (tid < length) {
        if (message[tid] == 'a') {
            message[tid] = 'z';
        } else {
            message[tid] = (message[tid] - 1);
        }
    }
}

int main() {
    char message[] = "hundreddaysofgpu";
    int length = sizeof(message) - 1;
    char *d_message;
    cudaError_t err;
    cudaEvent_t start, stop;
    float elapsedTime;
    err = cudaEventCreate(&start);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error creating start event: %s\n", cudaGetErrorString(err));
        return 1;
    }
    err = cudaEventCreate(&stop);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error creating stop event: %s\n", cudaGetErrorString(err));
        return 1;
    }
    err = cudaMalloc((void**)&d_message, length * sizeof(char));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error allocating device memory: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaEventRecord(start, 0);
    err = cudaMemcpy(d_message, message, length * sizeof(char), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error copying data to device: %s\n", cudaGetErrorString(err));
        return 1;
    }
    dim3 block(8, 8, 1);
    dim3 grid(1, 1, 1);
    encryptKernel<<<grid, block>>>(d_message, length);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error launching kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    err = cudaMemcpy(message, d_message, length * sizeof(char), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error copying data to host: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Encrypted message: %s\n", message);
    printf("Time taken for encryption: %.2f ms\n", elapsedTime);
    printf("Bandwidth Utilization = %.2fKB/s\n", ((float)length * sizeof(char) * 2) / (elapsedTime * 1e3));
    cudaEventRecord(start, 0);
    err = cudaMemcpy(d_message, message, length * sizeof(char), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error copying data to device: %s\n", cudaGetErrorString(err));
        return 1;
    }
    decryptKernel<<<grid, block>>>(d_message, length);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error launching kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    err = cudaMemcpy(message, d_message, length * sizeof(char), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error copying data to host: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Decrypted message: %s\n", message);
    printf("Time taken for decryption: %.2f ms\n", elapsedTime);
    printf("Bandwidth Utilization = %.2fKB/s\n", ((float)length * sizeof(char) * 2) / (elapsedTime * 1e3));
    err = cudaFree(d_message);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error freeing device memory: %s\n", cudaGetErrorString(err));
        return 1;
    }
    err = cudaEventDestroy(start);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error destroying start event: %s\n", cudaGetErrorString(err));
        return 1;
    }
    err = cudaEventDestroy(stop);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error destroying stop event: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}