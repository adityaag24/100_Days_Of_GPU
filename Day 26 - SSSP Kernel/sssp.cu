#include <stdio.h>
#include <cuda.h>
#include <assert.h>
#include "graphRep.cuh"
#include <time.h>
#define TPB 32
__global__ void initDistanceKernel(Graph *g, int source){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < g->V) {
        g->d[tid] = (tid == source)?0: 10000000000.0;
    }
}
static int s2i(const char *s){
    char *p; intmax_t r;
    errno = 0;
    r = strtoimax(s, &p, 10);
    if (errno != 0 || *p != '\0' || r <= 0 || r >= INT_MAX) {
        BAIL("s2i(\"%s\") -> %" PRIdMAX ", errno => %s\n", s, r, ERRSTR);
    }
    return (int)r;
}
int main(int argc, char **argv) {
    int a, b, line = 0, t = 0;
    cudaEvent_t start;
    cudaEvent_t stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    float wt;
    FILE *fp;
    Graph *h_G = new Graph;
    Graph *g_G = new Graph;
    if (argc != 4)
        BAIL("Usage: %s V E edgeListFile\n", argv[0]);
    h_G->setVertices(s2i(argv[1]));
    h_G->setEdges(s2i(argv[2]));
    if ((fp = fopen(argv[3], "r")) == NULL) 
        BAIL("fopen(%s) = %s", argv[3], ERRSTR);
    //Allocate Memory for the CSR Structure
    CALI(h_G->N, 2 + (size_t)(h_G->V), sizeof(*(h_G->N)));
    CALI(h_G->F, 0 + (size_t)(h_G->E), sizeof(*(h_G->F)));
    CALF(h_G->W, 0 + (size_t)(h_G->E), sizeof(*(h_G->W)));
    CALI(h_G->d, 0 + (size_t)(h_G->V), sizeof(*(h_G->d)));
    while (fscanf(fp,"%d %d %f", &a, &b, &wt) == 3) {
        line++;
        if (a > h_G->V || b > h_G->V)
            BAIL("%d: Bad Vertex Id: %d,%d\n", line, a, b);
        if (a == b){
            fprintf(stderr, "Line %d Same Vertex = %d\n",line, a);
        }
        h_G->N[a]++;
    }
    if (!feof(fp))
        BAIL("Parse Error after %d lines: %s\n", line, ERRSTR);
    if (line != h_G->E) 
        BAIL("Number of Edges (%d) is not same as the lines in the input file (%d)", h_G->E, line);
    for (a = 0; a <= h_G->V; a++) {
        t        += h_G->N[a];
        h_G->N[a]  = t;
    }
    assert(h_G->N[h_G->V] == h_G->E);
    rewind(fp);
    while (fscanf(fp,"%d %d %f", &a, &b, &wt) == 3) {
        int idx = --(h_G->N[a]);
        h_G->F[idx] = b;
        h_G->W[idx] = wt;
    }
    if (fclose(fp) != 0)
        BAIL("fclose():%s\n", ERRSTR);
    Graph *d_G;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_G,                       sizeof(Graph)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&(d_G->N), ( 2 + h_G->V ) * sizeof(int  )));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&(d_G->F), ( 0 + h_G->E ) * sizeof(int  )));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&(d_G->W), ( 0 + h_G->E ) * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&(d_G->d), ( 0 + h_G->V ) * sizeof(int  )));

    //Copy the Graph Class to Copy Vertices/Edges
    CHECK_CUDA_ERROR(cudaMemcpy(d_G,    h_G,                   sizeof(Graph), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_G->N, h_G->N, (2 + h_G->V) * sizeof(int),   cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_G->F, h_G->F, (0 + h_G->E) * sizeof(int),   cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_G->W, h_G->W, (0 + h_G->E) * sizeof(float), cudaMemcpyHostToDevice));

    //Initialize the Distance Vector by calling a kernel.
    int nBlocks = (h_G->V + TPB - 1) / TPB;
    int source  = 0;
    initDistanceKernel<<<nBlocks, TPB>>>(d_G, source);

    //Call the SSSP Kernel for the distance calculation
    
}