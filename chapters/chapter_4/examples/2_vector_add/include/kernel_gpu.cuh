#pragma once

#include <cuda_runtime.h>

__global__ void vectorAddKernel(float *A, float *B, float *C, int N);

void vectorAddKernel_launch(float *A, float *B, float *C, int N, int blocksPerGrid, int threadsPerBlock);