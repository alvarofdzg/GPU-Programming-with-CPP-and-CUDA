#include "kernel_gpu.cuh"


__global__ void checkPrimeGPU(long long start, long long end) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    long long num = start + (tid * 2);
    bool isPrime = true;
    if (num <= 1) {
        isPrime = false;
        return;
    }
    if (num == 2) {
        isPrime = true;
        return;
    } 
    if (num % 2 == 0) {
        isPrime = false;
        return;
    }
    if (num > end) return;

    
    for (long long i = 3; i * i <= num; i += 2) {
        if (num % i == 0) {
            isPrime = false;
            break;
        }
    }
}


void checkPrimeGPU_launch(long long start, long long end, int blocksPerGrid, int threadsPerBlock) {
    checkPrimeGPU<<<blocksPerGrid, threadsPerBlock>>>(start, end);
}