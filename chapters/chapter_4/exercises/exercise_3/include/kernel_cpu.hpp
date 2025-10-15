#pragma once

#include <vector>
#include <cmath>
#include <algorithm>


void vectorMultiplicationCPU(
    const std::vector<float>& vector_a, 
    const std::vector<float>&  vector_b, 
    std::vector<float>& vector_result, 
    int totalNumbers
);

void vectorDivisionCPU(
    const std::vector<float>& vector_a, 
    const std::vector<float>&  vector_b, 
    std::vector<float>& vector_result, 
    int totalNumbers
);

void vectorAbsoluteDifferenceCPU(
    const std::vector<float>& vector_a, 
    const std::vector<float>&  vector_b, 
    std::vector<float>& vector_result, 
    int totalNumbers
);

void vectorMaximumCPU(
    const std::vector<float>& vector_a, 
    const std::vector<float>&  vector_b, 
    std::vector<float>& vector_result, 
    int totalNumbers
);

void vectorMinimumCPU(
    const std::vector<float>& vector_a, 
    const std::vector<float>&  vector_b, 
    std::vector<float>& vector_result, 
    int totalNumbers
);

void vectorModuleCPU(
    const std::vector<float>& vector_a, 
    const std::vector<float>&  vector_b, 
    std::vector<float>& vector_result, 
    int totalNumbers
);

