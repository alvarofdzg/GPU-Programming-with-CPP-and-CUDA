#include <chrono>
#include <iostream>

#include "common_values.hpp"
#include "kernel_cpu.hpp"
#include "kernel_gpu.cuh"


int main() {
    int numPoints = 10'000'000;
    size_t sizePoints = numPoints * sizeof(Point);
    size_t sizeDistances = numPoints * sizeof(float);

    Point *h_lineA = (Point *)malloc(sizePoints);
    Point *h_lineB = (Point *)malloc(sizePoints);
    float *h_distances = (float *)malloc(sizeDistances);

    for (int i = 0; i < numPoints; i++) {
        h_lineA[i].x = i * 1.0f;
        h_lineA[i].y = i * 2.0f;
        h_lineA[i].z = i * 3.0f;
        h_lineB[i].x = i * 0.5f;
        h_lineB[i].y = i * 1.5f;
        h_lineB[i].z = i * 2.5f;
    }

    Point *d_lineA;
    Point *d_lineB;
    float *d_distances;
    cudaMalloc((void **)&d_lineA, sizePoints);
    cudaMalloc((void **)&d_lineB, sizePoints);
    cudaMalloc((void **)&d_distances, sizeDistances);
    
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaEventRecord(startEvent);
    cudaMemcpy(d_lineA, h_lineA, sizePoints, cudaMemcpyHostToDevice);
    cudaMemcpy(d_lineB, h_lineB, sizePoints, cudaMemcpyHostToDevice);
    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);

    float gpuCopyTime = 0;
    cudaEventElapsedTime(&gpuCopyTime, startEvent, stopEvent);
    std::cout<< std::fixed << "Time to copy data to GPU: " << gpuCopyTime << " ms" << std::endl;

    int blockSize = 1024;
    int gridSize = (numPoints + blockSize - 1) / blockSize;

    cudaEventRecord(startEvent);
    calculateEuclideanDistanceGPU_launch(d_lineA, d_lineB, d_distances, numPoints, gridSize, blockSize);
    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);
    
    float gpuExecutionTime = 0;
    cudaEventElapsedTime(&gpuExecutionTime, startEvent, stopEvent);
    std::cout<< std::fixed << "Time taken: " << gpuExecutionTime << " ms" << std::endl;
    
    cudaEventRecord(startEvent);
    cudaMemcpy(h_distances, d_distances, sizeDistances, cudaMemcpyDeviceToHost);
    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);

    float gpuRetrieveTime = 0;
    cudaEventElapsedTime(&gpuRetrieveTime, startEvent, stopEvent);
    std::cout<< std::fixed << "Time taken to copy results back GPU: " << gpuRetrieveTime << " ms" << std::endl << std::endl;
    float gpuDuration = (gpuCopyTime + gpuExecutionTime + gpuRetrieveTime);
    std::cout << "Time taken by GPU: " << gpuDuration << " ms" << std::endl;


    auto start = std::chrono::high_resolution_clock::now();
    calculateEuclideanDistanceCPU(h_lineA, h_lineB, h_distances, numPoints);
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpuDuration = (stop - start);

    std::cout << "Time taken by CPU: " << cpuDuration.count() << " ms" << std::endl;
    std::cout << "========================================== " << std::endl;

    std::cout << "speed up (execution time only): " << cpuDuration.count() / gpuExecutionTime << std::endl;
    std::cout << "speed up (GPU total time): " << cpuDuration.count() / gpuDuration << std::endl;

    cudaFree(d_lineA);
    cudaFree(d_lineB);
    cudaFree(d_distances);

    free(h_lineA);
    free(h_lineB);
    free(h_distances);

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
}