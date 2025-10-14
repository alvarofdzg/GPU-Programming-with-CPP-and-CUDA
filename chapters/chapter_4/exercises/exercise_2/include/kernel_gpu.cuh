#pragma once

#include <vector>

#include <cuda_runtime.h>

__global__ void scalarMultiplicationGPU(float number, const float* d_array, float* d_result, int totalNumbers);

void scalarMultiplicationGPU_launch(
    float number, 
    const float* d_array, 
    float* d_result, 
    int totalNumbers, 
    int blocksPerGrid, 
    int threadsPerBlock
);