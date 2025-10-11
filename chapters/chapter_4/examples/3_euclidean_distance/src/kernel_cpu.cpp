#include "kernel_cpu.hpp"

void calculateEuclideanDistanceCPU(Point *lineA, Point *lineB, float *distances, int numPoints) {
    for (int idx = 0; idx < numPoints; idx++) {
        float dx = lineA[idx].x - lineB[idx].x;
        float dy = lineA[idx].y - lineB[idx].y;
        float dz = lineA[idx].z - lineB[idx].z;
        
        distances[idx] = sqrt(dx * dx + dy * dy + dz * dz);
    }
}