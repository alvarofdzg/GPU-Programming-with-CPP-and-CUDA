#pragma once

#include <cmath>
#include <vector>
#include <cuda_runtime.h>

__global__ void vectorExponentGPU(const float* d_input_array, float* d_vector_result, int totalNumbers);
void vectorExponentGPU_launch(
    const float* d_input_array,
    float* d_vector_result, 
    int totalNumbers, 
    int blocksPerGrid, 
    int threadsPerBlock
);

__global__ void vectorSquareRootGPU(const float* d_input_array, float* d_vector_result, int totalNumbers);
void vectorSquareRootGPU_launch(
    const float* d_input_array,
    float* d_vector_result, 
    int totalNumbers, 
    int blocksPerGrid, 
    int threadsPerBlock
);