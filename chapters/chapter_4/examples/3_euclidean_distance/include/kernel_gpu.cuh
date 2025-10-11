#pragma once

#include "common_values.hpp"

#include <cuda_runtime.h>

__global__ void calculateEuclideanDistanceGPU(Point *lineA, Point *lineB, float *distances, int numPoints);

void calculateEuclideanDistanceGPU_launch(Point *lineA, Point *lineB, float *distances, int numPoints, int blocksPerGrid, int threadsPerBlock);