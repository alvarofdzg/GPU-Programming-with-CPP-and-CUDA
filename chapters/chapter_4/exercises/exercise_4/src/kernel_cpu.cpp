#include "kernel_cpu.hpp"


void vectorExponentCPU(
    const std::vector<float>& input_array, 
    std::vector<float>& vector_result, 
    int totalNumbers
) {
    for (int i = 0; i < totalNumbers; ++i) {
        vector_result[i] = pow(input_array[i], 2);
    }
}


void vectorSquareRootCPU(
    const std::vector<float>& input_array, 
    std::vector<float>& vector_result, 
    int totalNumbers
) {
    for (int i = 0; i < totalNumbers; ++i) {
        vector_result[i] = sqrt(input_array[i]);
    }
}