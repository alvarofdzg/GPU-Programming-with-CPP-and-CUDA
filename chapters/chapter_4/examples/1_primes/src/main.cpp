#include <chrono>
#include <iostream>

#include "kernel_cpu.hpp"
#include "kernel_gpu.cuh"

int main() {
    long long start =  100'001LL; // must start with odd
    long long end   =  190'001LL;

    int threadsPerBlock = 256;
    int totalNumbers = (end - start) / 2 + 1;
    int blocksPerGrid = (totalNumbers + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaEventRecord(startEvent, 0);

    checkPrimeGPU_launch(start, end, blocksPerGrid, threadsPerBlock);

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    float gpuDuration = 0;
    cudaEventElapsedTime(&gpuDuration, startEvent, stopEvent);

    std::cout << "Time taken on GPU: " << gpuDuration << " ms" << std::endl;

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    auto startTime = std::chrono::high_resolution_clock::now();
    for (long long num = start; num <= end; num += 2) {
        checkPrimeCPU(num);
    }
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpuDuration = endTime - startTime;

    std::cout << "Time taken on CPU: " << std::fixed << cpuDuration.count() << " ms" << std::endl;
    std::cout << "speed up : " << cpuDuration.count() / gpuDuration << std::endl;
}