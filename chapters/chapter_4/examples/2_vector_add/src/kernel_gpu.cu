#include "kernel_gpu.cuh"


__global__ void vectorAddGPU(float *A, float *B, float *C, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}


void vectorAddGPU_launch(float *A, float *B, float *C, int N, int blocksPerGrid, int threadsPerBlock) {
    vectorAddGPU<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
}