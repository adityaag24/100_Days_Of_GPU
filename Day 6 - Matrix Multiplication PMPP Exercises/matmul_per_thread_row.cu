#include <stdio.h>
#include <cuda.h>
#define TPB_x 32
#define r_A 4096
#define c_A 4096
#define r_B 4096
#define c_B 4096
// Error checking macro
#define CHECK_CUDA_ERROR(call) \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s, line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}
__host__ void printMatrix(int *a, int r, int c){
    for(int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++){
            printf("%d ", a[i*c + j]);
        }
        printf("\n");
    }
}
__host__ void matmulHost(int *a, int *b, int *c, int m, int k, int n){
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            int sum = 0;
            for (int l = 0; l < k; l++){
                sum += (a[i * k + l]*b[l * n + j]);
            }
            c[i * n + j] = sum;
        }
    }
}
__global__ void matmul(int *d_A, int *d_B, int *d_C) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < r_A) {
        for (int j = 0; j < c_B; j++) {
            int pVal = 0;
            for (int l = 0; l < c_A; l++){
                pVal += (d_A[row * c_A + l] * d_B[l * c_B + j]);
            }
            d_C[row * c_B + j] = pVal;
        }
    }
}
int main() {
    int *h_A, *h_B, *h_C;
    int *d_A, *d_B, *d_C;
    cudaEvent_t start, stop;
    float timeTaken;
    h_A = (int *)malloc(r_A * c_A * sizeof(int));
    h_B = (int *)malloc(r_B * c_B * sizeof(int));
    h_C = (int *)malloc(r_A * c_B * sizeof(int));
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_A, r_A * c_A * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_B, r_B * c_B * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_C, r_A * c_B * sizeof(int)));
    for (int i = 0; i < r_A; i++){
        for (int j = 0; j < c_A; j++) {
            h_A[i * c_A + j] = rand() % 10;
        }
    }
    for (int i = 0; i < r_B; i++){
        for (int j = 0; j < c_B; j++){
            h_B[i * c_B + j] = rand() % 10;
        }
    }
    dim3 g((r_A + TPB_x - 1)/TPB_x, 1, 1);
    dim3 b(TPB_x, 1, 1);
    cudaEventRecord(start, 0);
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, r_A * c_A * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, r_B * c_B * sizeof(int), cudaMemcpyHostToDevice));
    matmul<<<g, b>>>(d_A, d_B, d_C);
    CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, r_A * c_B * sizeof(int), cudaMemcpyDeviceToHost));
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&timeTaken, start, stop));
    printf("Time Taken = %.2f ms\n", timeTaken);
    // printMatrix(h_A, r_A, c_A);
    // printMatrix(h_B, r_B, c_B);
    // printMatrix(h_C, r_A, c_B);
    int *v_C = (int *)malloc(r_A * c_B * sizeof(int));
    matmulHost(h_A, h_B, v_C, r_A, c_A, c_B);
#ifdef VERIFY
    for (int i = 0; i < r_A; i++){
        for (int j = 0; j < c_B; j++){
            if (v_C[i * c_B + j] != h_C[i * c_B + j]){
                printf("Output Differs for (%d,%d) = %d vs %d\n", i, j, v_C[i * c_B + j], h_C[i * c_B + j]);
                return 0;
            }
        }
    }
    printf("Output Verified \n");
#endif
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
}