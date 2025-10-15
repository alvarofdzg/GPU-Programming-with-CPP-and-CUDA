#include "kernel_cpu.hpp"


void vectorMultiplicationCPU(
    const std::vector<float>& vector_a, 
    const std::vector<float>& vector_b, 
    std::vector<float>& vector_result, 
    int totalNumbers
) {
    for (int i = 0; i < totalNumbers; ++i) {
        vector_result[i] = vector_a[i] * vector_b[i];
    }
}


void vectorDivisionCPU(
    const std::vector<float>& vector_a, 
    const std::vector<float>& vector_b, 
    std::vector<float>& vector_result, 
    int totalNumbers
) {
    for (int i = 0; i < totalNumbers; ++i) {
        vector_result[i] = vector_a[i] / vector_b[i];
    }
}


void vectorAbsoluteDifferenceCPU(
    const std::vector<float>& vector_a, 
    const std::vector<float>& vector_b, 
    std::vector<float>& vector_result, 
    int totalNumbers
) {
    for (int i = 0; i < totalNumbers; ++i) {
        vector_result[i] = std::abs(vector_a[i] - vector_b[i]);
    }
}


void vectorMaximumCPU(
    const std::vector<float>& vector_a, 
    const std::vector<float>& vector_b, 
    std::vector<float>& vector_result, 
    int totalNumbers
) {
    for (int i = 0; i < totalNumbers; ++i) {
        vector_result[i] = std::max(vector_a[i], vector_b[i]);
    }
}


void vectorMinimumCPU(
    const std::vector<float>& vector_a, 
    const std::vector<float>& vector_b, 
    std::vector<float>& vector_result, 
    int totalNumbers
) {
    for (int i = 0; i < totalNumbers; ++i) {
        vector_result[i] = std::min(vector_a[i], vector_b[i]);
    }
}


void vectorModuleCPU(
    const std::vector<float>& vector_a, 
    const std::vector<float>& vector_b, 
    std::vector<float>& vector_result, 
    int totalNumbers
) {
    for (int i = 0; i < totalNumbers; ++i) {
        vector_result[i] = (vector_b[i] != 0.0f) ? std::fmod(vector_a[i], vector_b[i]) : std::numeric_limits<float>::quiet_NaN();
    }
}