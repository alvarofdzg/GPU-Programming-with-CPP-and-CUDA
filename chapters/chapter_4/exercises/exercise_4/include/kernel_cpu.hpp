#pragma once

#include <vector>
#include <cmath>

void vectorExponentCPU(
    const std::vector<float>& input_array, 
    std::vector<float>& vector_result, 
    int totalNumbers
);

void vectorSquareRootCPU(
    const std::vector<float>& input_array, 
    std::vector<float>& vector_result, 
    int totalNumbers
);

