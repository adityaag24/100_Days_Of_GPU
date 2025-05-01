#include <cuda.h>
#include <stdio.h>
#define TPB_x 32
#define TPB_y 32
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_image_write.h"

// Error checking macro
#define CHECK_CUDA_ERROR(call) \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s, line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

__global__ void colorToGrayscale(unsigned char *input_image, unsigned char *output_image, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        // Calculate linear position in input (RGB) array
        int input_idx = (y * width + x) * channels;
        
        // Calculate linear position in output (grayscale) array
        int output_idx = y * width + x;
        
        unsigned char r = input_image[input_idx];
        unsigned char g = input_image[input_idx + 1];
        unsigned char b = input_image[input_idx + 2];
        
        // Convert to grayscale using luminance formula
        unsigned char l = 0.21f * r + 0.71f * g + 0.07f * b;
        
        output_image[output_idx] = l;
    }
}

int main() {
    int width, height, channels;
    unsigned char *d_iImage = NULL;
    unsigned char *d_oImage = NULL;
    unsigned char *o_image  = NULL;
    
    unsigned char *image = stbi_load("sky.jpg", &width, &height, &channels, 0);
    if (image == NULL) {
        printf("Unable to load image\n");
        return 0;
    }
    
    printf("Loaded Image of width %d, height %d and channels %d\n", width, height, channels);
    size_t iImageSize = width * height * channels * sizeof(unsigned char);
    size_t oImageSize = width * height * 1        * sizeof(unsigned char);
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_iImage, iImageSize));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_oImage, oImageSize));
    o_image = (unsigned char *)malloc(oImageSize);
    if (o_image == NULL) {
        printf("Failed to allocate host memory\n");
        return 1;
    }
    CHECK_CUDA_ERROR(cudaMemcpy(d_iImage, image, iImageSize, cudaMemcpyHostToDevice));
    
    // Calculate grid dimensions to cover the entire image
    dim3 blockSize(TPB_x, TPB_y);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                  (height + blockSize.y - 1) / blockSize.y);
    
    colorToGrayscale<<<gridSize, blockSize>>>(d_iImage, d_oImage, width, height, channels);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaMemcpy(o_image, d_oImage, oImageSize, cudaMemcpyDeviceToHost));
    
    // Write PNG file with correct parameters
    // For grayscale, stride is just width (channels=1)
    if (!stbi_write_png("output.png", width, height, 1, o_image, width)) {
        printf("Failed to write output image\n");
    } else {
        printf("Successfully wrote output.png\n");
    }
    stbi_image_free(image);
    free(o_image);
    cudaFree(d_iImage);
    cudaFree(d_oImage);
    
    return 0;
}