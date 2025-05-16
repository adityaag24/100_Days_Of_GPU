#include <cuda.h>
#include <stdio.h>
// Error checking macro
#define CHECK_CUDA_ERROR(call) \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s, line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}
#define TPB 32
#define TILE_WIDTH 32
#define r_A 4096
#define c_A 4096
#define r_B 4096
#define c_B 4096
__global__ void matmul(int *a, int *b, int *c){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if (row < r_A && col < c_B) {
        for (int i = 0; i < r_B; i++) {
            sum += (a[row * r_B + i] * b[i * c_B + col]);
        }
        c[row * c_B + col] = sum; 
    }
}
__global__ void matmulTiling(int *a, int *b, int *c) {
    __shared__ int Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ int Nds[TILE_WIDTH][TILE_WIDTH];
    int bx  = blockIdx.x;  int by = blockIdx.y;
    int tx  = threadIdx.x; int ty = threadIdx.y;
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;
    int val = 0;
    for (int m = 0; m < c_B/TILE_WIDTH; m++) {
        Mds[ty][tx] = a[row * c_B + m * TILE_WIDTH + tx ];
        Nds[ty][tx] = b[(m * TILE_WIDTH + ty)*c_B  + col];
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; k++){
            val += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }
    c[row * c_B + col] = val;
}
__host__ void printMatrix(int *a, int r, int c){
    for(int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++){
            printf("%d ", a[i*c + j]);
        }
        printf("\n");
    }
}
__host__ void matmulHost(int *a, int *b, int *c){
    for (int i = 0; i < r_A; i++){
        for (int j = 0; j < c_B; j++){
            int sum = 0;
            for (int l = 0; l < c_A; l++){
                sum += (a[i * c_A + l]*b[l * c_B + j]);
            }
            c[i * c_B + j] = sum;
        }
    }
}
int main() {
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time_ms = 0.0f;
    int *h_a, *h_b, *h_c;
    int *d_a, *d_b, *d_c;
    h_a = (int *)malloc(r_A * c_A * sizeof(int));
    h_b = (int *)malloc(r_B * c_B * sizeof(int));
    h_c = (int *)malloc(r_A * c_B * sizeof(int));
    cudaMalloc((void **)&d_a, r_A * c_A * sizeof(int));
    cudaMalloc((void **)&d_b, r_B * c_B * sizeof(int));
    cudaMalloc((void **)&d_c, r_A * c_B * sizeof(int));
    for (int i = 0; i < r_A; i++){
        for (int j = 0; j < c_A; j++) {
            h_a[i * c_B + j] = rand() % 10;
        }
    }
    for (int i = 0; i < r_B; i++) {
        for (int j = 0; j < c_B; j++) {
            h_b[i * c_B + j] = rand() % 10;
        }
    }
    cudaMemcpy(d_a, h_a, r_A * c_A * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, r_B * c_B * sizeof(int), cudaMemcpyHostToDevice);
    cudaEventRecord(start, 0);
    dim3 grid((c_B + TPB - 1) / TPB, (r_A + TPB - 1) / TPB, 1);
    dim3 block(TPB, TPB, 1);
    matmulTiling<<<grid, block>>>(d_a, d_b, d_c);
    cudaEventRecord(stop,  0);
    cudaEventSynchronize(stop);
    cudaMemcpy(h_c, d_c, r_A * c_B * sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventElapsedTime(&time_ms, start, stop);
    // printMatrix(h_a, r_A, c_A);
    // printMatrix(h_b, r_B, c_B);
    // printMatrix(h_c, r_A, c_B);
    printf("Time Taken = %0.2f ms\n", time_ms);
    int *v_c = (int *)malloc(r_A * c_B * sizeof(int));
    matmulHost(h_a,h_b,v_c);
    for (int i = 0; i < r_A; i++){
        for (int j = 0; j < c_B; j++){
            if (v_c[i * c_B + j] != h_c[i * c_B + j]){
                printf("Output Differs for (%d,%d) = %d vs %d\n", i, j, v_c[i * c_B + j], h_c[i * c_B + j]);
                return 0;
            }
        }
    }
    printf("Output Verified\n");
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    free(v_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}