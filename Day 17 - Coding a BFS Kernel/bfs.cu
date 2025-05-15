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
__global__ void BFSGPU(int *d_N, int *d_F, int *d_V, int *d_E, float *d_W, float *cost, bool *done, bool *frontier, bool *visited) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < *d_V) {
        if (visited[tid] == false && frontier[tid] == true) {
            visited[tid]  = true;
            frontier[tid] = false;
            __syncthreads();
            int sIdx = d_N[tid+0];
            int eIdx = d_N[tid+1];
            for (int nid = d_F[sIdx];nid < d_F[eIdx]; nid++){
                if (visited[nid] == false) {
                    cost[nid]     = cost[tid] + 1;
                    frontier[nid] = true;
                    *done         = false;
                }
            }
        } else {
            *done = true;
        }
    } else {
        *done = false;
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
    int minVertex = INT_MAX;
    rewind(fp);
    while (fscanf(fp,"%d %d %f", &a, &b, &wt) == 3) {
        int idx = --csr.N[a];
        csr.F[idx] = b;
        csr.W[idx] = wt;
        minVertex = min(minVertex, a);
        minVertex = min(minVertex, b);
    }
    if (fclose(fp) != 0)
        BAIL("fclose():%s\n", ERRSTR);
    int *d_N, *d_F, *d_V, *d_E;
    float *d_W;
    float *d_cost,     *h_cost;
    bool  *d_done,      h_done;
    bool  *d_visited,  *h_visited;
    bool  *d_frontier, *h_frontier;
    CALF(h_cost,    0 + (size_t)csr.V, sizeof(*h_cost));
    CALB(h_visited, 0 + (size_t)csr.V, sizeof(*h_visited));
    CALB(h_frontier,0 + (size_t)csr.V, sizeof(*h_frontier));
    h_frontier[minVertex] = true;
    CHECK_CUDA_ERROR(cudaEventRecord(start, 0));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_V, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_E, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_N, (csr.V + 1) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_F, (csr.E + 0) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_W, (csr.E + 0) * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_done, sizeof(bool)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_cost, csr.V * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_visited, csr.V * sizeof(bool)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_frontier, csr.V * sizeof(bool))); 
    CHECK_CUDA_ERROR(cudaMemcpy(d_V, &(csr.V), sizeof(csr.V), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_E, &(csr.E), sizeof(csr.E), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_N, (csr.N), (csr.V + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_F, (csr.F), (csr.E + 0) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_W, (csr.W), (csr.E + 0) * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_cost, h_cost, csr.V * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_visited, h_visited, csr.V * sizeof(bool), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_frontier, h_frontier, csr.V * sizeof(bool), cudaMemcpyHostToDevice));
    int nBlocks = (csr.V + TPB - 1) / TPB;
    int count = 0;
    do {
        count++;
        h_done = true;
        CHECK_CUDA_ERROR(cudaMemcpy(d_done, &h_done, sizeof(bool), cudaMemcpyHostToDevice));
        BFSGPU<<<nBlocks, TPB>>>(d_N, d_F, d_V, d_E, d_W, d_cost, d_done, d_frontier, d_visited);
        CHECK_CUDA_ERROR(cudaMemcpy(&h_done, d_done, sizeof(bool), cudaMemcpyDeviceToHost));
    } while (!h_done);
    CHECK_CUDA_ERROR(cudaEventRecord(stop, 0));
    cudaEventSynchronize(stop);
    float timeTaken;
    cudaEventElapsedTime(&timeTaken, start, stop);
    printf("Time Taken by GPU = %0.2f ms\n", timeTaken);
    printf("BFS Took %d kernel calls\n", count);
    cudaFree(d_E);
    cudaFree(d_V);
    cudaFree(d_N);
    cudaFree(d_W);
    cudaFree(d_F);
}