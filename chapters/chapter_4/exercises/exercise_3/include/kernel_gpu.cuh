#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>

__global__ void vectorMultiplicationGPU(const float* d_vector_a, const float* d_vector_b, float* d_vector_result, int totalNumbers);
void vectorMultiplicationGPU_launch(
    const float* d_vector_a, 
    const float* d_vector_b, 
    float* d_vector_result, 
    int totalNumbers, 
    int blocksPerGrid, 
    int threadsPerBlock
);

__global__ void vectorDivisionGPU(const float* d_vector_a, const float* d_vector_b, float* d_vector_result, int totalNumbers);
void vectorDivisionGPU_launch(
    const float* d_vector_a, 
    const float* d_vector_b, 
    float* d_vector_result, 
    int totalNumbers, 
    int blocksPerGrid, 
    int threadsPerBlock
);

__global__ void vectorAbsoluteDifferenceGPU(const float* d_vector_a, const float* d_vector_b, float* d_vector_result, int totalNumbers);
void vectorAbsoluteDifferenceGPU_launch(
    const float* d_vector_a, 
    const float* d_vector_b, 
    float* d_vector_result, 
    int totalNumbers, 
    int blocksPerGrid, 
    int threadsPerBlock
);

__global__ void vectorMaximumGPU(const float* d_vector_a, const float* d_vector_b, float* d_vector_result, int totalNumbers);
void vectorMaximumGPU_launch(
    const float* d_vector_a, 
    const float* d_vector_b, 
    float* d_vector_result, 
    int totalNumbers, 
    int blocksPerGrid, 
    int threadsPerBlock
);

__global__ void vectorMinimumGPU(const float* d_vector_a, const float* d_vector_b, float* d_vector_result, int totalNumbers);
void vectorMinimumGPU_launch(
    const float* d_vector_a, 
    const float* d_vector_b, 
    float* d_vector_result, 
    int totalNumbers, 
    int blocksPerGrid, 
    int threadsPerBlock
);

__global__ void vectorModuleGPU(const float* d_vector_a, const float* d_vector_b, float* d_vector_result, int totalNumbers);
void vectorModuleGPU_launch(
    const float* d_vector_a, 
    const float* d_vector_b, 
    float* d_vector_result, 
    int totalNumbers, 
    int blocksPerGrid, 
    int threadsPerBlock
);