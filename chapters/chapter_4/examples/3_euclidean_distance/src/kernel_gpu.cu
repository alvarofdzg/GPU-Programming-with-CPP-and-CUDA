#include "kernel_gpu.cuh"


__global__ void calculateEuclideanDistanceGPU(Point *lineA, Point *lineB, float *distances, int numPoints) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < numPoints) {
        float dx = lineA[idx].x - lineB[idx].x;
        float dy = lineA[idx].y - lineB[idx].y;
        float dz = lineA[idx].z - lineB[idx].z;
        
        distances[idx] = sqrtf(dx * dx + dy * dy + dz * dz);
    }
}


void calculateEuclideanDistanceGPU_launch(Point *lineA, Point *lineB, float *distances, int numPoints, int blocksPerGrid, int threadsPerBlock) {
    calculateEuclideanDistanceGPU<<<blocksPerGrid, threadsPerBlock>>>(lineA, lineB, distances, numPoints);
}