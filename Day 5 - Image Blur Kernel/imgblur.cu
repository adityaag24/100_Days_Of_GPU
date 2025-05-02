#include <stdio.h>
#include <cuda.h>
#define BLUR_SIZE 7
#define TPB_x     32
#define TPB_y     32
#define R 0
#define G 1
#define B 2
#define A 3
// Error checking macro
#define CHECK_CUDA_ERROR(call) \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s, line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_image_write.h"
__global__ void imageBlur(unsigned char *inputImage, unsigned char *outputImage, int width, int height, int channels, int channel, int copyAlpha) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < height && col < width) {
        int pixVal = 0;
        int pixels = 0;
        if (copyAlpha) {
            outputImage[(row * width + col) * channels + A] = inputImage[(row * width + col) * channels + A];
        }
        for (int blurRow = -BLUR_SIZE; blurRow <= BLUR_SIZE; blurRow++) {
            for (int blurCol = -BLUR_SIZE; blurCol <= BLUR_SIZE; blurCol++) {
                int curRow = row + blurRow;
                int curCol = col + blurCol;
                
                if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width) {
                    pixVal += inputImage[(curRow * width + curCol ) * channels + channel];
                    pixels++;
                }
            }
        }
        outputImage[(row * width + col) * channels + channel] = (unsigned char)(pixVal / pixels);
    }
}
int main() {
    stbi_set_flip_vertically_on_load(1);
    unsigned char *h_inputImage  = NULL;
    unsigned char *h_outputImage = NULL;
    unsigned char *d_inputImage  = NULL;
    unsigned char *d_outputImage = NULL;
    
    int width, height, channels;
    h_inputImage = stbi_load("input.jpg", &width, &height, &channels, 0);
    if (!h_inputImage) {
        fprintf(stderr, "Unable to read image\n");
        return 0;
    }
    printf("Loaded Image of width %d, height %d and %d channels\n", width, height, channels);
    h_outputImage = (unsigned char *)malloc(width * height * channels * sizeof(unsigned char));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_inputImage, width * height * channels * sizeof(unsigned char)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_outputImage, width * height * channels * sizeof(unsigned char)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_inputImage, h_inputImage, width * height * channels * sizeof(unsigned char), cudaMemcpyHostToDevice));
    dim3 grid ((width + TPB_x - 1)/TPB_x, (height + TPB_y - 1)/TPB_y, 1);
    dim3 block(TPB_x,TPB_y,1);
    imageBlur<<<grid, block>>>(d_inputImage, d_outputImage, width, height, channels, R, 0);
    imageBlur<<<grid, block>>>(d_inputImage, d_outputImage, width, height, channels, G, 0);
    imageBlur<<<grid, block>>>(d_inputImage, d_outputImage, width, height, channels, B, 1);
    CHECK_CUDA_ERROR(cudaMemcpy(h_outputImage, d_outputImage, width * height * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    if (!stbi_write_png("output.png", width, height, channels, h_outputImage, width * channels)) {
        printf("Failed to write output image\n");
    } else {
        printf("Successfully wrote output.png\n");
    }
    stbi_image_free(h_inputImage);
    free(h_outputImage);
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
}