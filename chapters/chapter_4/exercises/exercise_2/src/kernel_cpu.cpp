#include "kernel_cpu.hpp"

void scalarMultiplicationCPU(float number, std::vector<float>& array, std::vector<float>& results, int totalNumbers) {
    for (int i = 0; i < totalNumbers; ++i) {
        results[i] = array[i] * number;
    }
}
