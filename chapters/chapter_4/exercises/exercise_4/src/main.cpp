/*
Now, with a single input array calculate and return the results on a different array:
- The exponentiation of the array elements to the power of 2
- The square root of the elements of the array
*/


#include <chrono>
#include <iostream>

#include "kernel_cpu.hpp"
#include "kernel_gpu.cuh"

void vectorExponent(
    const std::vector<float>& input_array, 
    const float* d_input_array, 
    int totalNumbers, 
    int blocksPerGrid, 
    int threadsPerBlock
) {
    float* d_vector_result = nullptr;
    cudaMalloc(&d_vector_result, totalNumbers * sizeof(float));
    std::vector<float> h_result_gpu(totalNumbers, 0.0f);
    std::vector<float> result_cpu(totalNumbers, 0.0f);

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaEventRecord(startEvent, 0);
    vectorExponentGPU_launch(d_input_array, d_vector_result, totalNumbers, blocksPerGrid, threadsPerBlock);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    float gpuDuration = 0;
    cudaEventElapsedTime(&gpuDuration, startEvent, stopEvent);
    std::cout << "Time taken on GPU: " << gpuDuration << " ms" << std::endl;

    // Copy results back
    cudaMemcpy(h_result_gpu.data(), d_vector_result,
                         totalNumbers * sizeof(float), cudaMemcpyDeviceToHost);
                         
    auto startTime = std::chrono::high_resolution_clock::now();
    vectorExponentCPU(input_array, result_cpu, totalNumbers);
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpuDuration = endTime - startTime;
    std::cout << "Time taken on CPU: " << std::fixed << cpuDuration.count() << " ms" << std::endl;
    std::cout << "speed up : " << cpuDuration.count() / gpuDuration << std::endl;
    
    for (int i = 0; i < totalNumbers; ++i) {
        if (h_result_gpu[i] != result_cpu[i]) {
            std::cout << "Mismatch in result at index " << i << ": GPU " << h_result_gpu[i] << " vs CPU " << result_cpu[i] << std::endl;
        }
    }
    
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaFree(d_vector_result);
}

void vectorSquareRoot(
    const std::vector<float>& input_array, 
    const float* d_input_array, 
    int totalNumbers, 
    int blocksPerGrid, 
    int threadsPerBlock
) {
    float* d_vector_result = nullptr;
    cudaMalloc(&d_vector_result, totalNumbers * sizeof(float));
    std::vector<float> h_result_gpu(totalNumbers, 0.0f);
    std::vector<float> result_cpu(totalNumbers, 0.0f);

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaEventRecord(startEvent, 0);
    vectorSquareRootGPU_launch(d_input_array, d_vector_result, totalNumbers, blocksPerGrid, threadsPerBlock);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    float gpuDuration = 0;
    cudaEventElapsedTime(&gpuDuration, startEvent, stopEvent);
    std::cout << "Time taken on GPU: " << gpuDuration << " ms" << std::endl;

    // Copy results back
    cudaMemcpy(h_result_gpu.data(), d_vector_result,
                         totalNumbers * sizeof(float), cudaMemcpyDeviceToHost);
                         
    auto startTime = std::chrono::high_resolution_clock::now();
    vectorSquareRootCPU(input_array, result_cpu, totalNumbers);
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpuDuration = endTime - startTime;
    std::cout << "Time taken on CPU: " << std::fixed << cpuDuration.count() << " ms" << std::endl;
    std::cout << "speed up : " << cpuDuration.count() / gpuDuration << std::endl;
    
    for (int i = 0; i < totalNumbers; ++i) {
        if (h_result_gpu[i] != result_cpu[i]) {
            std::cout << "Mismatch in result at index " << i << ": GPU " << h_result_gpu[i] << " vs CPU " << result_cpu[i] << std::endl;
        }
    }
    
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaFree(d_vector_result);
}


int main() {

    std::vector<float> input_array;
    int totalNumbers = 80000;
    for (int i = 2; i < totalNumbers; i++) {
        input_array.push_back(static_cast<float>(100001 + i * 2));
    }

    int threadsPerBlock = 256;
    int blocksPerGrid = (totalNumbers + threadsPerBlock - 1) / threadsPerBlock;

    float* d_input_array = nullptr;
    cudaMalloc(&d_input_array,  totalNumbers * sizeof(float));
    
    cudaMemcpy(d_input_array, input_array.data(),
                totalNumbers * sizeof(float), cudaMemcpyHostToDevice);

    // Launch vector exponent
    std::cout << "Vector exponent:" << std::endl;
    vectorExponent(input_array, d_input_array, totalNumbers, blocksPerGrid, threadsPerBlock);
    std::cout << "========================================== " << std::endl;
    std::cout << "\n";

    // Launch vector exponent
    std::cout << "Vector exponent:" << std::endl;
    vectorExponent(input_array, d_input_array, totalNumbers, blocksPerGrid, threadsPerBlock);
    std::cout << "========================================== " << std::endl;
    std::cout << "\n";

    // Launch vector exponent
    std::cout << "Vector square root:" << std::endl;
    vectorSquareRoot(input_array, d_input_array, totalNumbers, blocksPerGrid, threadsPerBlock);
    std::cout << "========================================== " << std::endl;
    std::cout << "\n";
}