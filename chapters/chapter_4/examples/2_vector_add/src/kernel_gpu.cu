#include "kernel_gpu.cuh"


__global__ void vectorAddKernel(float *A, float *B, float *C, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}


void vectorAddKernel_launch(float *A, float *B, float *C, int N, int blocksPerGrid, int threadsPerBlock) {
    vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
}