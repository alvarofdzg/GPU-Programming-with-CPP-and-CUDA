#include "kernel_gpu.cuh"


__global__ void scalarMultiplicationGPU(float number, const float* d_array, float* d_result, int totalNumbers) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < totalNumbers) {
        d_result[tid] = d_array[tid] * number;
    }
}


void scalarMultiplicationGPU_launch(float number, const float* d_array, float* d_result, int totalNumbers, int blocksPerGrid, int threadsPerBlock) {
    scalarMultiplicationGPU<<<blocksPerGrid, threadsPerBlock>>>(number, d_array, d_result, totalNumbers);
}