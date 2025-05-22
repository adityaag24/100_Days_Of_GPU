# 100_Days_Of_GPU
Code Repository for the 100 Days of GPU Challenge
## Day 1
> Coding: Vector Addition Kernel for a Data of 1024 Elements

> Reading: Read Chapter 1 of PMPP Book

## Day 2
> Coding:-

>> Encrypting and Decrypting a Character Array of lower-case letters.

>> Used the CUDA Events API to record the time and compute the bandwidth utilization.

> Reading: 

>> Read Chapter 2 of PMPP Book

>> Read about [Performance Metrics in CUDA](https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc)

## Day 3
> Coding:-

>> Coded the Matrix Multiplication Kernel Basic

> Reading:-

>> Read about Matmul from [this link](https://drive.google.com/file/d/1ZH4-TlBWy5c9JP7gXxNQ4dAKZBZVwvHY/view)

>> Solved the Exercises of Chapter 2 of PMPP book.

## Day 4
> Coding:-

>> Coded the Color to Grayscale Image Converter Kernel

> Reading:-

>> Read Chapter 3 (Half) of the PMPP book.

## Day 5
> Coding: -

>> Coded the Image Blur Kernel 

> Reading:-

>> Read about Image Blur Kernel from PMPP book.

## Day 6
> Coding: -

>> Coded the Matrix Multiplication Kernel that calculates the output with per thread activity: -

>>> Row-Wise

>>> Col-Wise

> Reading: -

>> Solved the Exercise of Chapter 3 in PMPP book.

## Day 7
> Coding: -

>> Coded the Matrix-Vector Multiplication Kernel

> Reading: -

>> Started Reading Chapter 4 of the PMPP Book.

## Day 8
> Coding: -

>> Coded the Reduction Kernel using Atomic Operation

> Reading: -

>> Read the Chapter 4 of the PMPP Book.

## Day 9
> Coding: -

>> Coded the Reduction Kernel that uses interleaved addressing

> Reading: -

>> Reading about reduction kernels from [this link](https://drive.google.com/file/d/18MxiGsXZ6QPj7q1r1RFkn2DoFHc7jNb6/view)

## Day 10
> Coding: -

>> Coded the Reduction Kernel that uses sequential addressing to reduce thread divergence.

> Reading: -

>> Reading about reduction kernels from [this link](https://drive.google.com/file/d/18MxiGsXZ6QPj7q1r1RFkn2DoFHc7jNb6/view)

## Day 11
> Coding: -

>> Coded the Reduction Kernel that uses sequenced addressing to reduce bank conflicts.

> Reading: -

>> Reading about reduction kernels from [this link](https://drive.google.com/file/d/18MxiGsXZ6QPj7q1r1RFkn2DoFHc7jNb6/view)

## Day 12
> Coding: -

>> Coded the Reduction Kernel that uses sequenced addressing to reduce bank conflicts and increases per thread activity by adding an element while loading

> Reading: -

>> Reading about reduction kernels from [this link](https://drive.google.com/file/d/18MxiGsXZ6QPj7q1r1RFkn2DoFHc7jNb6/view)

## Day 13
> Coding: -

>> Coded the Reduction Kernel that uses sequenced addressing to reduce bank conflicts and increases per thread activity by adding an element while loading. Also added a warp reduce optimization to optimize the execution for the last warp.

> Reading: -

>> Reading about reduction kernels from [this link](https://drive.google.com/file/d/18MxiGsXZ6QPj7q1r1RFkn2DoFHc7jNb6/view)

## Day 14
> Coding: -

>> Coded the CSR Representation in CPU but using the CSR to print the Degree of each Node using a CUDA Kernel.

> Reading: -

>> Reading about CSR Format from [this paper](https://www.usenix.org/system/files/login/articles/login_winter20_16_kelly.pdf)

## Day 15
> Coding: -

>> Added Edge Weights to the CSR Kernel

>> Added Timing for GPU/CPU Comparison

>> Used the Synthetic Benchmark Suite to Generate a Graph of 250k edges and 15k vertices and did performance measurements

> Reading: -

>> Reading about CSR Format from [this paper](https://www.usenix.org/system/files/login/articles/login_winter20_16_kelly.pdf)

>> [Synthetic Benchmark Suite](https://networkrepository.com/networks.php)

## Day 16
> Coding: -

>> Wrote a GPU Kernel to Print Per-Vertex Edge Weight Mean

>> Used the Synthetic Benchmark Suite to Generate a Graph of 250k edges and 15k vertices and did performance measurements: -

>> Command: ./csr 15229 245952 .\bio-CE-CX.edges

> Reading: -

>> Reading about CSR Format from [this paper](https://www.usenix.org/system/files/login/articles/login_winter20_16_kelly.pdf)

>> [Synthetic Benchmark Suite](https://networkrepository.com/networks.php)

## Day 17
> Coding: -

>> Wrote a GPU Kernel to Implement BFS

> Reading: -

>> Reading about BFS algorithm in [this paper](https://www.nvidia.co.uk/content/cudazone/CUDABrowser/downloads/Accelerate_Large_Graph_Algorithms/HiPC.pdf)

## Day 18
> Coding: -

>> Wrote a GPU Kernel to Implement Matrix Multiplication using Tiling

## Day 19
> Coding: -

>> Wrote a GPU Kernel to Invert Colors of an image

## Day 20
> Coding: -

>> Wrote a GPU Kernel to Compute the ReLU function for a set of floating point numbers.

## Day 21
> Coding: -

>> Wrote a GPU Kernel to Compute the Leaky ReLU function for a set of floating point numbers.

>> Performance Analysis vs CPU version

## Day 22
> Coding: -

>> Wrote a GPU Kernel to Compute the Matrix Transpose by loading through rows and storing by columns.

## Day 23
> Coding: -

>> Wrote a GPU Kernel to Compute the Matrix Transpose by loading through columns and storing by rows.