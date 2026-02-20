/*
Create a program based on the prime number testing program but updating two arrays: the first with
the number tested by the thread and the second, on the same index position, to hold the test result (whether
or not the number is prime). Copy the results back to the host and print only the prime numbers.
*/


#include <vector>
#include <cstdint>
#include <chrono>
#include <iostream>

#include "kernel_cpu.hpp"
#include "kernel_gpu.cuh"

int main() {
    long long start = 100'001LL; // must start with odd
    long long end = 190'001LL;

    int threadsPerBlock = 256;
    int totalNumbers = (end - start) / 2 + 1;
    int blocksPerGrid = (totalNumbers + threadsPerBlock - 1) / threadsPerBlock;

    std::vector<int> h_tested_gpu(totalNumbers, 0);
    std::vector<uint8_t> h_isPrime_gpu(totalNumbers, 0);
    std::vector<int> tested_cpu(totalNumbers, 0);
    std::vector<uint8_t> isPrime_cpu(totalNumbers, 0);

    // Device buffers
    int* d_tested = nullptr;
    uint8_t* d_isPrime = nullptr;
    cudaMalloc(&d_tested,  totalNumbers * sizeof(int));
    cudaMalloc(&d_isPrime, totalNumbers * sizeof(uint8_t));

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaEventRecord(startEvent);
    checkPrimeGPU_launch(start, end, totalNumbers, d_tested, d_isPrime, blocksPerGrid, threadsPerBlock);
    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);

    float gpuDuration = 0;
    cudaEventElapsedTime(&gpuDuration, startEvent, stopEvent);
    std::cout << "Time taken on GPU: " << gpuDuration << " ms" << std::endl;

    // Copy results back
    cudaMemcpy(h_tested_gpu.data(), d_tested,
                         totalNumbers * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_isPrime_gpu.data(), d_isPrime,
                         totalNumbers * sizeof(uint8_t), cudaMemcpyDeviceToHost);


    auto startTime = std::chrono::high_resolution_clock::now();
    checkPrimeCPULoop(start, end, tested_cpu, isPrime_cpu);
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpuDuration = endTime - startTime;
    std::cout << "Time taken on CPU: " << std::fixed << cpuDuration.count() << " ms" << std::endl;

    for (size_t i = 0; i < h_tested_gpu.size(); ++i) {
        if (h_tested_gpu[i] != tested_cpu[i]) {
            std::cout << "Mismatch in tested number at index " << i << ": GPU " << h_tested_gpu[i] << " vs CPU " << tested_cpu[i] << std::endl;
        }
        if (h_isPrime_gpu[i] != isPrime_cpu[i]) {
            std::cout << "Mismatch in prime result at index " << i << ": GPU " << (int)h_isPrime_gpu[i] << " vs CPU " << (int)isPrime_cpu[i] << std::endl;
        }
    }

    // Cleanup
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaFree(d_tested);
    cudaFree(d_isPrime);
}