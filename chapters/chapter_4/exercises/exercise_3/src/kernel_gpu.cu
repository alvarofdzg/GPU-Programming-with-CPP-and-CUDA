#include "kernel_gpu.cuh"


__global__ void vectorMultiplicationGPU(const float* d_vector_a, const float* d_vector_b, float* d_vector_result, int totalNumbers) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < totalNumbers) {
        d_vector_result[tid] = d_vector_a[tid] * d_vector_b[tid];
    }
}
void vectorMultiplicationGPU_launch(
    const float* d_vector_a, 
    const float* d_vector_b, 
    float* d_vector_result, 
    int totalNumbers, 
    int blocksPerGrid, 
    int threadsPerBlock
) {
    vectorMultiplicationGPU<<<blocksPerGrid, threadsPerBlock>>>(d_vector_a, d_vector_b, d_vector_result, totalNumbers);
}


__global__ void vectorDivisionGPU(const float* d_vector_a, const float* d_vector_b, float* d_vector_result, int totalNumbers) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < totalNumbers) {
        d_vector_result[tid] = d_vector_a[tid] / d_vector_b[tid];
    }
}
void vectorDivisionGPU_launch(
    const float* d_vector_a, 
    const float* d_vector_b, 
    float* d_vector_result, 
    int totalNumbers, 
    int blocksPerGrid, 
    int threadsPerBlock
) {
    vectorDivisionGPU<<<blocksPerGrid, threadsPerBlock>>>(d_vector_a, d_vector_b, d_vector_result, totalNumbers);
}


__global__ void vectorAbsoluteDifferenceGPU(const float* d_vector_a, const float* d_vector_b, float* d_vector_result, int totalNumbers) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < totalNumbers) {
        d_vector_result[tid] = std::abs(d_vector_a[tid] - d_vector_b[tid]);
    }
}
void vectorAbsoluteDifferenceGPU_launch(
    const float* d_vector_a, 
    const float* d_vector_b, 
    float* d_vector_result, 
    int totalNumbers, 
    int blocksPerGrid, 
    int threadsPerBlock
) {
    vectorAbsoluteDifferenceGPU<<<blocksPerGrid, threadsPerBlock>>>(d_vector_a, d_vector_b, d_vector_result, totalNumbers);
}


__global__ void vectorMaximumGPU(const float* d_vector_a, const float* d_vector_b, float* d_vector_result, int totalNumbers) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < totalNumbers) {
        d_vector_result[tid] = std::max(d_vector_a[tid], d_vector_b[tid]);
    }
}
void vectorMaximumGPU_launch(
    const float* d_vector_a, 
    const float* d_vector_b, 
    float* d_vector_result, 
    int totalNumbers, 
    int blocksPerGrid, 
    int threadsPerBlock
) {
    vectorMaximumGPU<<<blocksPerGrid, threadsPerBlock>>>(d_vector_a, d_vector_b, d_vector_result, totalNumbers);
}


__global__ void vectorMinimumGPU(const float* d_vector_a, const float* d_vector_b, float* d_vector_result, int totalNumbers) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < totalNumbers) {
        d_vector_result[tid] = std::min(d_vector_a[tid], d_vector_b[tid]);
    }
}
void vectorMinimumGPU_launch(
    const float* d_vector_a, 
    const float* d_vector_b, 
    float* d_vector_result, 
    int totalNumbers, 
    int blocksPerGrid, 
    int threadsPerBlock
) {
    vectorMinimumGPU<<<blocksPerGrid, threadsPerBlock>>>(d_vector_a, d_vector_b, d_vector_result, totalNumbers);
}


__global__ void vectorModuleGPU(const float* d_vector_a, const float* d_vector_b, float* d_vector_result, int totalNumbers) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < totalNumbers) {
        float denom = d_vector_b[tid];
        // define your policy for denom == 0 (NaN, 0, or keep a[i])
        d_vector_result[tid] = (denom != 0.0f) ? fmodf(d_vector_a[tid], denom) : NAN;  // or 0.0f
    }
}
void vectorModuleGPU_launch(
    const float* d_vector_a, 
    const float* d_vector_b, 
    float* d_vector_result, 
    int totalNumbers, 
    int blocksPerGrid, 
    int threadsPerBlock
) {
    vectorModuleGPU<<<blocksPerGrid, threadsPerBlock>>>(d_vector_a, d_vector_b, d_vector_result, totalNumbers);
}