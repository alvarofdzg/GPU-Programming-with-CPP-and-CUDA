/*
Now, with two input arrays calculate the element-wise (corresponding elements on each index):
- Vector multiplication
- Vector division
- Vector absolute difference
- Vector maximum - the result array should receive the maximum of the two input elements
- Vector minimum - the result array should receive the minimum of the two input elements
- The modules of the element from the first array and the second array
*/


#include <chrono>
#include <iostream>

#include "kernel_cpu.hpp"
#include "kernel_gpu.cuh"

void vectorMultiplication(
    const std::vector<float>& vector_a, 
    const std::vector<float>& vector_b, 
    const float* d_vector_a, 
    const float* d_vector_b, 
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
    vectorMultiplicationGPU_launch(d_vector_a, d_vector_b, d_vector_result, totalNumbers, blocksPerGrid, threadsPerBlock);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    float gpuDuration = 0;
    cudaEventElapsedTime(&gpuDuration, startEvent, stopEvent);
    std::cout << "Time taken on GPU: " << gpuDuration << " ms" << std::endl;

    // Copy results back
    cudaMemcpy(h_result_gpu.data(), d_vector_result,
                         totalNumbers * sizeof(float), cudaMemcpyDeviceToHost);
                         
    auto startTime = std::chrono::high_resolution_clock::now();
    vectorMultiplicationCPU(vector_a, vector_b, result_cpu, totalNumbers);
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

void vectorDivision(
    const std::vector<float>& vector_a, 
    const std::vector<float>& vector_b, 
    const float* d_vector_a, 
    const float* d_vector_b, 
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
    vectorDivisionGPU_launch(d_vector_a, d_vector_b, d_vector_result, totalNumbers, blocksPerGrid, threadsPerBlock);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    float gpuDuration = 0;
    cudaEventElapsedTime(&gpuDuration, startEvent, stopEvent);
    std::cout << "Time taken on GPU: " << gpuDuration << " ms" << std::endl;

    // Copy results back
    cudaMemcpy(h_result_gpu.data(), d_vector_result,
                         totalNumbers * sizeof(float), cudaMemcpyDeviceToHost);
                         
    auto startTime = std::chrono::high_resolution_clock::now();
    vectorDivisionCPU(vector_a, vector_b, result_cpu, totalNumbers);
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

void vectorAbsoluteDifference(
    const std::vector<float>& vector_a, 
    const std::vector<float>& vector_b, 
    const float* d_vector_a, 
    const float* d_vector_b, 
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
    vectorAbsoluteDifferenceGPU_launch(d_vector_a, d_vector_b, d_vector_result, totalNumbers, blocksPerGrid, threadsPerBlock);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    float gpuDuration = 0;
    cudaEventElapsedTime(&gpuDuration, startEvent, stopEvent);
    std::cout << "Time taken on GPU: " << gpuDuration << " ms" << std::endl;

    // Copy results back
    cudaMemcpy(h_result_gpu.data(), d_vector_result,
                         totalNumbers * sizeof(float), cudaMemcpyDeviceToHost);
                         
    auto startTime = std::chrono::high_resolution_clock::now();
    vectorAbsoluteDifferenceCPU(vector_a, vector_b, result_cpu, totalNumbers);
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

void vectorMaximum(
    const std::vector<float>& vector_a, 
    const std::vector<float>& vector_b, 
    const float* d_vector_a, 
    const float* d_vector_b, 
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
    vectorMaximumGPU_launch(d_vector_a, d_vector_b, d_vector_result, totalNumbers, blocksPerGrid, threadsPerBlock);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    float gpuDuration = 0;
    cudaEventElapsedTime(&gpuDuration, startEvent, stopEvent);
    std::cout << "Time taken on GPU: " << gpuDuration << " ms" << std::endl;

    // Copy results back
    cudaMemcpy(h_result_gpu.data(), d_vector_result,
                         totalNumbers * sizeof(float), cudaMemcpyDeviceToHost);
                         
    auto startTime = std::chrono::high_resolution_clock::now();
    vectorMaximumCPU(vector_a, vector_b, result_cpu, totalNumbers);
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

void vectorMinimum(
    const std::vector<float>& vector_a, 
    const std::vector<float>& vector_b, 
    const float* d_vector_a, 
    const float* d_vector_b, 
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
    vectorMinimumGPU_launch(d_vector_a, d_vector_b, d_vector_result, totalNumbers, blocksPerGrid, threadsPerBlock);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    float gpuDuration = 0;
    cudaEventElapsedTime(&gpuDuration, startEvent, stopEvent);
    std::cout << "Time taken on GPU: " << gpuDuration << " ms" << std::endl;

    // Copy results back
    cudaMemcpy(h_result_gpu.data(), d_vector_result,
                         totalNumbers * sizeof(float), cudaMemcpyDeviceToHost);
                         
    auto startTime = std::chrono::high_resolution_clock::now();
    vectorMinimumCPU(vector_a, vector_b, result_cpu, totalNumbers);
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

void vectorModule(
    const std::vector<float>& vector_a, 
    const std::vector<float>& vector_b, 
    const float* d_vector_a, 
    const float* d_vector_b, 
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
    vectorModuleGPU_launch(d_vector_a, d_vector_b, d_vector_result, totalNumbers, blocksPerGrid, threadsPerBlock);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    float gpuDuration = 0;
    cudaEventElapsedTime(&gpuDuration, startEvent, stopEvent);
    std::cout << "Time taken on GPU: " << gpuDuration << " ms" << std::endl;

    // Copy results back
    cudaMemcpy(h_result_gpu.data(), d_vector_result,
                         totalNumbers * sizeof(float), cudaMemcpyDeviceToHost);
                         
    auto startTime = std::chrono::high_resolution_clock::now();
    vectorModuleCPU(vector_a, vector_b, result_cpu, totalNumbers);
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
    // Prepare data
    std::vector<float> vector_a;
    std::vector<float> vector_b;
    int totalNumbers = 80000;
    for (int i = 2; i < totalNumbers; i++) {
        vector_a.push_back(static_cast<float>(100001 + i * 2));
        vector_b.push_back(static_cast<float>(i * 2));
    }

    int threadsPerBlock = 256;
    int blocksPerGrid = (totalNumbers + threadsPerBlock - 1) / threadsPerBlock;

    float* d_vector_a = nullptr;
    float* d_vector_b = nullptr;
    cudaMalloc(&d_vector_a,  totalNumbers * sizeof(float));
    cudaMalloc(&d_vector_b, totalNumbers * sizeof(float));
    
    cudaMemcpy(d_vector_a, vector_a.data(),
                totalNumbers * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector_b, vector_b.data(),
                totalNumbers * sizeof(float), cudaMemcpyHostToDevice);

    // Launch vector multiplication
    std::cout << "Vector multiplication:" << std::endl;
    vectorMultiplication(vector_a, vector_b, d_vector_a, d_vector_b, totalNumbers, blocksPerGrid, threadsPerBlock);
    std::cout << "========================================== " << std::endl;
    std::cout << "\n";

    // Launch vector multiplication
    std::cout << "Vector multiplication:" << std::endl;
    vectorMultiplication(vector_a, vector_b, d_vector_a, d_vector_b, totalNumbers, blocksPerGrid, threadsPerBlock);
    std::cout << "========================================== " << std::endl;
    std::cout << "\n";
    
    // Launch vector division
    std::cout << "Vector division:" << std::endl;
    vectorDivision(vector_a, vector_b, d_vector_a, d_vector_b, totalNumbers, threadsPerBlock, blocksPerGrid);
    std::cout << "========================================== " << std::endl;
    std::cout << "\n";

    // Launch vector absolute difference
    std::cout << "Vector absolute difference:" << std::endl;
    vectorAbsoluteDifference(vector_a, vector_b, d_vector_a, d_vector_b, totalNumbers, threadsPerBlock, blocksPerGrid);
    std::cout << "========================================== " << std::endl;
    std::cout << "\n";

    // Launch vector maximum
    std::cout << "Vector maximum:" << std::endl;
    vectorMaximum(vector_a, vector_b, d_vector_a, d_vector_b, totalNumbers, threadsPerBlock, blocksPerGrid);
    std::cout << "========================================== " << std::endl;
    std::cout << "\n";

    // Launch vector minimum
    std::cout << "Vector minimum:" << std::endl;
    vectorMinimum(vector_a, vector_b, d_vector_a, d_vector_b, totalNumbers, threadsPerBlock, blocksPerGrid);
    std::cout << "========================================== " << std::endl;
    std::cout << "\n";

    // Launch vector module
    std::cout << "Vector module:" << std::endl;
    vectorModule(vector_a, vector_b, d_vector_a, d_vector_b, totalNumbers, threadsPerBlock, blocksPerGrid);
    std::cout << "========================================== " << std::endl;
    std::cout << "\n";

    // Free device memory
    cudaFree(d_vector_a);
    cudaFree(d_vector_b);
}