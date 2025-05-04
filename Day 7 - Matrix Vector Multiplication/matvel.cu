#include <stdio.h>
#include <cuda.h>
#define TPB 128
#define N_R 4096
#define N_C 4096
// Error checking macro
#define CHECK_CUDA_ERROR(call) \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s, line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}
__global__ void matVec(int *opMat, int *ipMat, int *ipVec, int R, int C){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < R) {
        int sumVal  = 0;
        for (int i = 0; i < C; i++){
            sumVal += (ipMat[row * C + i] * ipVec[i]);
        }
        opMat[row] = sumVal;
    }
}
__host__ void matVecHost(int *opMat, int *ipMat, int *ipVec, int R, int C){
    for (int i = 0; i < R; i++){
        int sum = 0;
        for (int l = 0; l < C; l++){
            sum += (ipMat[i * C + l] * ipVec[l]);
        }
        opMat[i] = sum;
    }
}
int main(){
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    int *h_ipMat, *h_opMat, *h_ipVec;
    int *d_ipMat, *d_opMat, *d_ipVec;
    int R = N_R;
    int C = N_C;
    h_ipMat = (int *)malloc(R * C * sizeof(int)); //Input  Matrix = R * C
    h_ipVec = (int *)malloc(C * 1 * sizeof(int)); //Input  Vector = C * 1 
    h_opMat = (int *)malloc(R * 1 * sizeof(int)); //Output Matrix = R * 1
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_ipMat, R * C * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_ipVec, C * 1 * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_opMat, R * 1 * sizeof(int)));
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++){
            h_ipMat[i * C + j] = rand() % 10;
        }
    }
    for (int i = 0; i < C; i++) {
        h_ipVec[i] = rand() % 10;
    }
    CHECK_CUDA_ERROR(cudaEventRecord(start, 0));
    CHECK_CUDA_ERROR(cudaMemcpy(d_ipMat, h_ipMat, R * C * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_ipVec, h_ipVec, C * 1 * sizeof(int), cudaMemcpyHostToDevice));
    dim3 grid((R + TPB - 1) / TPB, 1, 1);
    dim3 block(TPB, 1, 1);
    matVec<<<grid, block>>>(d_opMat, d_ipMat, d_ipVec, R, C);
    CHECK_CUDA_ERROR(cudaEventRecord(stop, 0));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaMemcpy(h_opMat, d_opMat, R * 1 * sizeof(int), cudaMemcpyDeviceToHost));
    float time;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));
    printf("Time Taken = %f ms\n", time);
#ifdef VERIFY
    int *v_c = (int *)malloc(R * 1 * sizeof(int));
    matVecHost(h_ipMat, h_ipVec, v_c, R, C);
    for (int i = 0; i < R; i++){
        if (v_c[i] != h_opMat[i]){
            printf("Output differs for element (%d) = %d vs %d\n",i, v_c[i],h_opMat[i]);
            return 0;
        }
    }
    printf("Output Verified\n");
#endif
}