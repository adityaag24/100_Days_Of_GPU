#include <stdio.h>
#include <cuda.h>
#include <assert.h>
#include "csrRep.h"
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
__host__ void DegreeCPU(CSR h_csr) {
    for (int a = 1; a <= h_csr.V; a++){
        printf("Degree of Vertex %d = %d\n", a, h_csr.N[a+1] - h_csr.N[a]);
    }
}
__global__ void printDegreeGPU(int *d_N, int *d_F, int *d_V, int *d_E) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x + 1;
    if (tid <= *d_V) {
        printf("Degree of Vertex %d = %d\n", tid, d_N[tid + 1] - d_N[tid]);
    }
}
int main(int argc, char **argv) {
    int a, b, line = 0, t = 0;
    FILE *fp;
    CSR csr;
    if (argc != 4)
        BAIL("Usage: %s V E edgeListFile\n", argv[0]);
    csr.V = s2i(argv[1]);
    csr.E = s2i(argv[2]);
    if ((fp = fopen(argv[3], "r")) == NULL) 
        BAIL("fopen(%s) = %s", argv[3], ERRSTR);
    //Allocate Memory for the CSR Structure
    CAL(csr.N, 2 + (size_t)csr.V, sizeof(*csr.N));
    CAL(csr.F, 0 + (size_t)csr.E, sizeof(*csr.F));
    while (fscanf(fp,"%d %d", &a, &b) == 2) {
        line++;
        if (a <= 0 || a > csr.V || b <= 0 || b > csr.V)
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
    for (a = 1; a <= csr.V; a++) {
        t        += csr.N[a];
        csr.N[a]  = t;
    }
    csr.N[a] = t;
    assert(csr.N[csr.V + 1] == csr.E);
    rewind(fp);
    while (fscanf(fp,"%d %d", &a, &b) == 2) {
        // printf("Reading Edge : (%d, %d)\n", a, b);
        csr.F[--csr.N[a]] = b;
    }
    if (fclose(fp) != 0)
        BAIL("fclose():%s\n", ERRSTR);
    int *d_N, *d_F, *d_V, *d_E;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_V, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_E, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_N, (csr.V + 2) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_F, (csr.E + 0) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_V, &(csr.V), sizeof(csr.V), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_E, &(csr.E), sizeof(csr.E), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_N, (csr.N), (csr.V + 2) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_F, (csr.F), (csr.E + 0) * sizeof(int), cudaMemcpyHostToDevice));
    int nBlocks = (csr.V + 1 + TPB - 1) / TPB;
    cudaError_t err;
    printDegreeGPU<<<nBlocks, TPB>>>(d_N, d_F, d_V, d_E);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err!=cudaSuccess) {
        fprintf(stderr, "%s\n", cudaGetErrorString(err));
    }
}