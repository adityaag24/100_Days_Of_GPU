#include <errno.h>
#include <stdint.h>
#include <inttypes.h>
// Error checking macro
#define CHECK_CUDA_ERROR(call) \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s, line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}
static_assert(sizeof(intmax_t) > sizeof(int), "int sizes"); // intmax_t is defined in stdint.h
static int s2i(const char *s);
#define ERRSTR strerror(errno)                             // Prints the Error Number with errno defined in errno.h
#define S1(s) #s                                           // Prints the name of the variable s
#define S2(s) S1(s)                                        // Calls S1 that prints the name of the variable s
#define COORDS __FILE__ ":" S2(__LINE__) ": "              // 
#define BAIL(...)                                         \
 do { fprintf(stderr, COORDS __VA_ARGS__);                \
      exit(EXIT_FAILURE); } while(0)
#define CALI(p, n, s) \
 do { (p) = (int *)calloc((n), (s)); if (!p)              \
        BAIL("calloc(%lu, %lu): %s\n", (n), (s), ERRSTR); \
    } while(0)
#define CALF(p, n, s) \
 do { (p) = (float *)calloc((n), (s)); if (!p)              \
        BAIL("calloc(%lu, %lu): %s\n", (n), (s), ERRSTR); \
    } while(0)
typedef struct CSRRec {
    int    V; //# of Vertices
    int    E; //# of Edges
    int   *N; //Vertex Index Array
    int   *F; //Edge         Array
    float *W; //Weight       Array
}CSR;
__host__   void printAdjacenciesCPU(CSR *h_csr, char **adjacencyList);
__global__ void printAdjacenciesGPU(CSR *d_csr, char **adjacencyList);