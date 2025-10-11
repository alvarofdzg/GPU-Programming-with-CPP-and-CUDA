/*
Create a program based on the prime number testing program but updating two arrays: the first with
the number tested by the thread and the second, on the same index position, to hold the test result (whether
or not the number is prime). Copy the results back to the host and print only the prime numbers.
*/


#include <vector>
#include <cstdint>
#include <chrono>
#include <iostream>

#include "kernel_gpu.cuh"

int main() {
    long long start = 100'001LL; // must start with odd
    long long end = 190'001LL;

    int threadsPerBlock = 256;
    int totalNumbers = (end - start) / 2 + 1;
    int blocksPerGrid = (totalNumbers + threadsPerBlock - 1) / threadsPerBlock;

    std::vector<int> h_tested(totalNumbers, 0);
    std::vector<uint8_t> h_isPrime(totalNumbers, 0);

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
    cudaMemcpy(h_tested.data(), d_tested,
                         totalNumbers * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_isPrime.data(), d_isPrime,
                         totalNumbers * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // Print primes
    std::cout << "Primes in [" << start << ", " << end << "]:\n";
    int printed = 0;
    for (int i = 0; i < totalNumbers; ++i) {
        if (h_isPrime[i]) {
            std::cout << h_tested[i] << ' ';
            if (++printed % 16 == 0) std::cout << '\n';
        }
    }
    if (printed % 16) std::cout << '\n';

    // Cleanup
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaFree(d_tested);
    cudaFree(d_isPrime);
}