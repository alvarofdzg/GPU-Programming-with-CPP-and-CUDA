#include "kernel_cpu.hpp"

uint8_t checkPrimeCPU(long long number) {
    if (number <= 1) return 0;
    if (number == 2) return 1;
    if (number % 2 == 0) return 0;
    for (long long i = 3; i * i <= number; i += 2) {
        if (number % i == 0) {
            return 0;
        }
    }
    return 1;
}

void checkPrimeCPULoop(long long start, long long end, std::vector<int>& tested, std::vector<uint8_t>& isPrime) {
    int count = 0;
    for (long long num = start; num <= end; num += 2) {
        tested[count] = num;
        isPrime[count] = checkPrimeCPU(num);
        count = count + 1;
    }
}