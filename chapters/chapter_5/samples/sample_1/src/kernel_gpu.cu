#include "kernel_gpu.cuh"

__global__ void checkPrimeGPU(
    long long start, 
    long long end, 
    int totalNumbers, 
    int* d_tested, 
    uint8_t* d_isPrime
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    long long num = start + (tid * 2);
    
    d_tested[tid] = static_cast<int>(num);
    
    bool isPrime = 1;
    if (num > end) return;   

    if (num <= 1) {
        isPrime = 0;
        return;
    }
    if (num == 2) {
        isPrime = 1;
        return;
    } 
    if (num % 2 == 0) {
        isPrime = 0;
        return;
    }
    for (long long i = 3; i * i <= num; i += 2) {
        if (num % i == 0) {
            isPrime = 0;
            break;
        }
    }
    d_isPrime[tid] = isPrime;
}


void checkPrimeGPU_launch(
    long long start, 
    long long end, 
    int totalNumbers, 
    int* d_tested, 
    uint8_t* d_isPrime, 
    int blocksPerGrid, 
    int threadsPerBlock
) {
    checkPrimeGPU<<<blocksPerGrid, threadsPerBlock>>>(start, end, totalNumbers, d_tested, d_isPrime);
}