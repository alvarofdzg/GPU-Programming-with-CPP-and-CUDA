/*
Create a program that performs the scalar multiplication of a vector. It receives as input parameters 
a float number and a float array, and executes the multiplcation of each element of the array by the 
given number. Remember to copy the result back to the host.
*/


#include <chrono>
#include <iostream>

#include "kernel_cpu.hpp"
#include "kernel_gpu.cuh"

int main() {
    // Prepare data
    float number = 5.0f;
    std::vector<float> array;
    for (size_t i = 0; i < 45000; i++) {
        array.push_back(static_cast<float>(100001 + i * 2));
    }

    int threadsPerBlock = 256;
    int totalNumbers = static_cast<int>(array.size());
    int blocksPerGrid = (totalNumbers + threadsPerBlock - 1) / threadsPerBlock;

    std::vector<float> h_result_gpu(totalNumbers, 0.0f);
    float* d_array = nullptr;
    float* d_result = nullptr;
    cudaMalloc(&d_array,  totalNumbers * sizeof(float));
    cudaMalloc(&d_result, totalNumbers * sizeof(float));

    cudaMemcpy(d_array, array.data(),
                totalNumbers * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaEventRecord(startEvent, 0);
    scalarMultiplicationGPU_launch(number, d_array, d_result, totalNumbers, blocksPerGrid, threadsPerBlock);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    float gpuDuration = 0;
    cudaEventElapsedTime(&gpuDuration, startEvent, stopEvent);
    std::cout << "Time taken on GPU: " << gpuDuration << " ms" << std::endl;

    // Copy results back
    cudaMemcpy(h_result_gpu.data(), d_result,
                         totalNumbers * sizeof(int), cudaMemcpyDeviceToHost);
                         
    std::vector<float> h_result_cpu(totalNumbers, 0.0f);
    auto startTime = std::chrono::high_resolution_clock::now();
    scalarMultiplicationCPU(number, array, h_result_cpu, totalNumbers);
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpuDuration = endTime - startTime;
    std::cout << "Time taken on CPU: " << std::fixed << cpuDuration.count() << " ms" << std::endl;
    std::cout << "speed up : " << cpuDuration.count() / gpuDuration << std::endl;
    
    for (int i = 0; i < totalNumbers; ++i) {
        if (h_result_gpu[i] != h_result_cpu[i]) {
            std::cout << "Mismatch in result at index " << i << ": GPU " << h_result_gpu[i] << " vs CPU " << h_result_cpu[i] << std::endl;
        }
    }
    
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaFree(d_array);
    cudaFree(d_result);
}