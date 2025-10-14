#pragma once

#include <vector>
#include <cstdint>

uint8_t checkPrimeCPU(long long number);
void checkPrimeCPULoop(long long start, long long end, std::vector<int>& tested, std::vector<uint8_t>& isPrime);