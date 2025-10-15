#include "kernel_gpu.cuh"


__global__ void vectorExponentGPU(const float* d_input_array, float* d_vector_result, int totalNumbers) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < totalNumbers) {
        d_vector_result[tid] = pow(d_input_array[tid], 2);
    }
}
void vectorExponentGPU_launch(
    const float* d_input_array, 
    float* d_vector_result, 
    int totalNumbers, 
    int blocksPerGrid, 
    int threadsPerBlock
) {
    vectorExponentGPU<<<blocksPerGrid, threadsPerBlock>>>(d_input_array, d_vector_result, totalNumbers);
}


__global__ void vectorSquareRootGPU(const float* d_input_array, float* d_vector_result, int totalNumbers) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < totalNumbers) {
        d_vector_result[tid] = sqrt(d_input_array[tid]);
    }
}
void vectorSquareRootGPU_launch(
    const float* d_input_array, 
    float* d_vector_result, 
    int totalNumbers, 
    int blocksPerGrid, 
    int threadsPerBlock
) {
    vectorSquareRootGPU<<<blocksPerGrid, threadsPerBlock>>>(d_input_array, d_vector_result, totalNumbers);
}