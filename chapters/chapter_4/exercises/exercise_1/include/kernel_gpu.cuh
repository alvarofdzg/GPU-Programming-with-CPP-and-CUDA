#pragma once

#include <cstdint>

#include <cuda_runtime.h>

__global__ void checkPrimeGPU(
    long long start, 
    long long end, 
    int totalNumbers, 
    int* d_tested, 
    uint8_t* d_isPrime
);

void checkPrimeGPU_launch(
    long long start, 
    long long end,  
    int totalNumbers, 
    int* d_tested, 
    uint8_t* d_isPrime, 
    int blocksPerGrid, 
    int threadsPerBlock
);