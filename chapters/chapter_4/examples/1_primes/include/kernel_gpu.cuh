#pragma once

#include <cuda_runtime.h>

__global__ void checkPrimeGPU(long long start, long long end);

void checkPrimeGPU_launch(long long start, long long end, int blocksPerGrid, int threadsPerBlock);