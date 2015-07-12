#include <iostream>
#include <iomanip>
#include <fstream>

#include <cuda.h>
#include <cuda_runtime.h>

#include "Matrix.h"
#include "GpuTimer.h"

__global__ void fivePoint(float const* input, float* output, size_t width, size_t height) {
	size_t x = blockIdx.x * blockDim.x + threadIdx.x;
	size_t y = blockIdx.y * blockDim.y + threadIdx.y;
	size_t index = y * width + x;

	size_t left = index - 1;
	size_t right = index + 1;
	size_t top = index - blockDim.y;
	size_t bottom = index + blockDim.y;

	if (x < 1 || y < 1 || x == width - 1 || y == height - 1) {
		output[index] = input[index];
	}
	else {
		output[index] = 1.0f / 5 * (input[left] + input[index] + input[right] + input[top] + input[bottom]);
	}
}

size_t const width = 1024, height = 1024;

int main() {
	float* input, * output;
	cudaMalloc(&input, sizeof(float) * width * height);
	cudaMalloc(&output, sizeof(float) * width * height);

	Matrix<float> data(width, height);

	cudaMemcpy(input, data.raw(), sizeof(float) * width * height, cudaMemcpyHostToDevice);

	dim3 blockSize { width, height, 1 };

	GpuTimer timer;

	timer.start();

	for (auto i = 0; i < 20; ++i) {
		fivePoint<<<1, blockSize>>>(input, output, width, height);
		fivePoint<<<1, blockSize>>>(output, input, width, height);
	}

	cudaMemcpy(data.raw(), output, sizeof(float) * width * height, cudaMemcpyDeviceToHost);

	timer.stop();

	std::ofstream file("file.txt");
	file << data;
	file.close();

	std::cout << "\nDauer: " << timer.getDuration().count() << " us" << std::endl;
}

