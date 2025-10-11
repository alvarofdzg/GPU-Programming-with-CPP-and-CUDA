#pragma once

#include <cuda_runtime.h>

__global__ void vectorAddGPU(float *A, float *B, float *C, int N);

void vectorAddGPU_launch(float *A, float *B, float *C, int N, int blocksPerGrid, int threadsPerBlock);