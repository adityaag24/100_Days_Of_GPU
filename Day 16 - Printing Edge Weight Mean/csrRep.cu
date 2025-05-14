#include <stdio.h>
#include <cuda.h>
#include <assert.h>
#include "csrRep.h"
#include <time.h>
#define TPB 16
static int s2i(const char *s){
    char *p; intmax_t r;
    errno = 0;
    r = strtoimax(s, &p, 10);
    if (errno != 0 || *p != '\0' || r <= 0 || r >= INT_MAX) {
        BAIL("s2i(\"%s\") -> %" PRIdMAX ", errno => %s\n", s, r, ERRSTR);
    }
    return (int)r;
}
__host__   void findMeanEdgeWeightCPU(CSR h_csr, float *h_m_weight) {
    for (int a = 0; a < h_csr.V; a++){
        int sIdx  = h_csr.N[a+0];
        int eIdx  = h_csr.N[a+1];
        int nIdx  = eIdx - sIdx;
        float avg = 0.0;
        for (int b = h_csr.F[sIdx]; b < h_csr.F[eIdx]; b++){
            avg = ((avg + h_csr.W[b]) / nIdx);
        }
        h_m_weight[a] = avg;
    }
}
__global__ void findMeanEdgeWeightGPU(int *d_N, int *d_F, int *d_V, int *d_E, float *d_W, float *d_m_weight) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < *d_V) {
        int sIdx  = d_N[tid + 0];
        int eIdx  = d_N[tid + 1];
        int nIdx  = eIdx - sIdx;
        float avg = 0.0;
        for (int b = d_F[sIdx]; b < d_F[eIdx]; b++){
            avg = ((avg + d_W[b]) / nIdx);
        }
        d_m_weight[tid] = avg;
    }
}
int main(int argc, char **argv) {
    int a, b, line = 0, t = 0;
    cudaEvent_t start;
    cudaEvent_t stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    float wt;
    FILE *fp;
    CSR csr;
    if (argc != 4)
        BAIL("Usage: %s V E edgeListFile\n", argv[0]);
    csr.V = s2i(argv[1]);
    csr.E = s2i(argv[2]);
    if ((fp = fopen(argv[3], "r")) == NULL) 
        BAIL("fopen(%s) = %s", argv[3], ERRSTR);
    //Allocate Memory for the CSR Structure
    CALI(csr.N, 2 + (size_t)csr.V, sizeof(*csr.N));
    CALI(csr.F, 0 + (size_t)csr.E, sizeof(*csr.F));
    CALF(csr.W, 0 + (size_t)csr.E, sizeof(*csr.W));
    while (fscanf(fp,"%d %d %f", &a, &b, &wt) == 3) {
        line++;
        if (a > csr.V || b > csr.V)
            BAIL("%d: Bad Vertex Id: %d,%d\n", line, a, b);
        if (a == b){
            fprintf(stderr, "Line %d Same Vertex = %d\n",line, a);
        }
        csr.N[a]++;
    }
    if (!feof(fp))
        BAIL("Parse Error after %d lines: %s\n", line, ERRSTR);
    if (line != csr.E) 
        BAIL("Number of Edges (%d) is not same as the lines in the input file (%d)", csr.E, line);
    for (a = 0; a <= csr.V; a++) {
        t        += csr.N[a];
        csr.N[a]  = t;
    }
    assert(csr.N[csr.V] == csr.E);
    rewind(fp);
    while (fscanf(fp,"%d %d %f", &a, &b, &wt) == 3) {
        int idx = --csr.N[a];
        csr.F[idx] = b;
        csr.W[idx] = wt;
    }
    if (fclose(fp) != 0)
        BAIL("fclose():%s\n", ERRSTR);
    int *d_N, *d_F, *d_V, *d_E;
    float *d_W;
    float *h_m_weight;
    float *d_h_m_weight;
    float *d_m_weight;
    CALF(h_m_weight,   (size_t)csr.V, sizeof(*h_m_weight));
    CALF(d_h_m_weight, (size_t)csr.V, sizeof(*d_h_m_weight));
    clock_t hostT;
    hostT = clock();
    findMeanEdgeWeightCPU(csr, h_m_weight);
    hostT = clock() - hostT;
    double hostTime = ( ((double)hostT)/CLOCKS_PER_SEC ) * 1e3;
    CHECK_CUDA_ERROR(cudaEventRecord(start, 0));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_V, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_E, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_N, (csr.V + 1) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_F, (csr.E + 0) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_W, (csr.E + 0) * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_m_weight, (csr.V + 0) * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_V, &(csr.V), sizeof(csr.V), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_E, &(csr.E), sizeof(csr.E), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_N, (csr.N), (csr.V + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_F, (csr.F), (csr.E + 0) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_W, (csr.W), (csr.E + 0) * sizeof(float), cudaMemcpyHostToDevice));
    int nBlocks = (csr.V + TPB - 1) / TPB;
    cudaError_t err;
    findMeanEdgeWeightGPU<<<nBlocks, TPB>>>(d_N, d_F, d_V, d_E, d_W, d_m_weight);
    CHECK_CUDA_ERROR(cudaEventRecord(stop, 0));
    cudaEventSynchronize(stop);
    err = cudaGetLastError();
    if (err!=cudaSuccess) {
        fprintf(stderr, "%s\n", cudaGetErrorString(err));
    } else {
        CHECK_CUDA_ERROR(cudaMemcpy(d_h_m_weight, d_m_weight, csr.V * sizeof(float), cudaMemcpyDeviceToHost));
        float timeTaken;
        for (int i = 0; i < csr.V; i++){
            if (h_m_weight[i] != d_h_m_weight[i]){
                printf("Output Differs for %d vertex = %f vs %f\n", d_h_m_weight[i], h_m_weight[i]);
                return -1;
            }
        }
        printf("Output Verified.\n");
        cudaEventElapsedTime(&timeTaken, start, stop);
        printf("Time Taken by CPU = %0.2lf ms\n", hostTime);
        printf("Time Taken by GPU = %0.2f ms\n", timeTaken);
        printf("Speedup Obtained  = %d\n", (int) (hostTime/timeTaken));
    }
    cudaFree(d_E);
    cudaFree(d_V);
    cudaFree(d_N);
    cudaFree(d_W);
    cudaFree(d_F);
    cudaFree(d_m_weight);
}