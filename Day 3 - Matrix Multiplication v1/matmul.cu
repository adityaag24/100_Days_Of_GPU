#include <cuda.h>
#include <stdio.h>
#define TPB 32
__global__ void matmul(int *a, int *b, int *c, int m, int k, int n){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if (row < m && col < n) {
        for (int i = 0; i < k; i++) {
            sum += (a[row * k + i] * b[i * n + col]);
        }
        c[row * n + col] = sum; 
    }
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
int main() {
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time_ms = 0.0f;
    int *h_a, *h_b, *h_c;
    int *d_a, *d_b, *d_c;
    int m, k, n;
    scanf("%d %d %d ",&m, &k, &n);
    h_a = (int *)malloc(m * k * sizeof(int));
    h_b = (int *)malloc(k * n * sizeof(int));
    h_c = (int *)malloc(m * n * sizeof(int));
    cudaMalloc((void **)&d_a, m * k * sizeof(int));
    cudaMalloc((void **)&d_b, k * n * sizeof(int));
    cudaMalloc((void **)&d_c, m * n * sizeof(int));
    for (int i = 0; i < m; i++){
        for (int j = 0; j < k; j++) {
            h_a[i * k + j] = rand() % 10;
        }
    }
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            h_b[i * n + j] = rand() % 10;
        }
    }
    cudaMemcpy(d_a, h_a, m * k * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, k * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaEventRecord(start, 0);
    dim3 grid((n + TPB - 1) / TPB, (m + TPB - 1) / TPB, 1);
    dim3 block(TPB, TPB, 1);
    matmul<<<grid, block>>>(d_a, d_b, d_c, m, k, n);
    cudaEventRecord(stop,  0);
    cudaEventSynchronize(stop);
    cudaMemcpy(h_c, d_c, m * n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventElapsedTime(&time_ms, start, stop);
    printf("===Matrix A===\n");
    printMatrix(h_a, m, k);
    printf("===Matrix B===\n");
    printMatrix(h_b, k ,n);
    printf("===Matrix C===\n");
    printMatrix(h_c, m, n);
    printf("Time Taken = %0.2f ms\n", time_ms);
    int *v_c = (int *)malloc(m * n * sizeof(int));
    matmulHost(h_a,h_b,v_c,m,k,n);
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            if (v_c[i * n + j] != h_c[i * n + j]){
                printf("Output Differs for (%d,%d) = %d vs %d\n", i, j, v_c[i*n+j], h_c[i*n+j]);
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