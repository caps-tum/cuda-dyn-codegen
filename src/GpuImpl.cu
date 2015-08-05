#include <iostream>
#include <iomanip>
#include <fstream>
#include <array>

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/device_new.h>
#include <thrust/device_delete.h>

#include "Matrix.h"
#include "GpuTimer.h"
#include "Stencil.h"
#include "Logger.h"

__global__ void fivePoint1(thrust::device_ptr<float const> const input, thrust::device_ptr<float> const output) {
	size_t x = blockIdx.x * blockDim.x + threadIdx.x;
	size_t y = blockIdx.y * blockDim.y + threadIdx.y;
	size_t width = gridDim.x * blockDim.x;
	size_t height = gridDim.y * blockDim.y;
	size_t index = y * width + x;

	size_t left = index - 1;
	size_t right = index + 1;
	size_t top = index - width;
	size_t bottom = index + width;

	if (x < 1 || y < 1 || x == width - 1 || y == height - 1) {
		output[index] = input[index];
	}
	else {
		output[index] = 1.0f / 5 * (input[left] + input[index] + input[right] + input[top] + input[bottom]);
	}
}

__global__ void fivePoint2(float const* input, float* output) {
	size_t x = blockIdx.x * blockDim.x + threadIdx.x + 1;
	size_t y = blockIdx.y * blockDim.y + threadIdx.y + 1;
	size_t width = gridDim.x * blockDim.x;
	size_t height = gridDim.y * blockDim.y;
	
	size_t index = y * width + x;

	size_t left = index - 1;
	size_t right = index + 1;
	size_t top = index - width;
	size_t bottom = index + width;

	output[index] = 1.0f / 5 * (input[left] + input[index] + input[right] + input[top] + input[bottom]);
}

__global__ void fivePoint_shared(float const* input, float* output) {
	size_t x = blockIdx.x * blockDim.x + threadIdx.x;
	size_t y = blockIdx.y * blockDim.y + threadIdx.y;
	size_t width = gridDim.x * blockDim.x;
	size_t height = gridDim.y * blockDim.y;
	size_t index = y * width + x;

	__shared__ float sharedbuffer[34 * 34];

	//if (threadIdx.x < warpSize) {
	//	for (auto i = threadIdx; i < 32; ++i) {
	//		sharedbuffer[i][threadIdx] = input[index];
	//	}
	//}
	
	int localIndex = threadIdx.y * blockDim.y + threadIdx.x;


	__syncthreads();

	size_t left = localIndex - 1;
	size_t right = localIndex + 1;
	size_t top = localIndex - blockDim.y;
	size_t bottom = localIndex + blockDim.y;

	if (x < 1 || y < 1 || x == width - 1 || y == height - 1) {
		output[index] = sharedbuffer[localIndex];
	}
	else {
		output[index] = 1.0f / 5 * (sharedbuffer[left] + sharedbuffer[localIndex] + sharedbuffer[right] + sharedbuffer[top] + sharedbuffer[bottom]);
	}
}

std::array<int, 5> sizes { 128, 256, 512, 1024, 2048 };
std::array<size_t, 5> results;

size_t iterationsPerSize = 60;

void test1() {
	// jeweils 2x
	for (auto k = 0; k < 2; ++k) {
		// für alle Größen
		for (auto s = 0; s < sizes.size(); ++s) {
			auto size = sizes[s];
			
			auto input = thrust::device_new<float>(size * size);
			auto output = thrust::device_new<float>(size * size);

			Matrix<float> data(size, size);

			thrust::copy_n(data.raw(), size * size, input);

			dim3 blockSize { 32, 32, 1 };
			dim3 gridSize { size / blockSize.x, size / blockSize.y, 1 };

			GpuTimer timer;

			timer.start();

			for (auto i = 0; i < iterationsPerSize / 2; ++i) {
				fivePoint1<<<gridSize, blockSize>>>(thrust::device_ptr<float const>(input), thrust::device_ptr<float>(output));
				fivePoint1<<<gridSize, blockSize>>>(thrust::device_ptr<float const>(output), thrust::device_ptr<float>(input));
			}

			timer.stop();

			results[s] += stencilsPerSecond(size, size, timer.getDuration()) / iterationsPerSize;

			thrust::copy_n(input, size * size, data.raw());

			thrust::device_delete(output, size * size);
			thrust::device_delete(input, size * size);			
		}
	}

	std::transform(std::begin(results), std::end(results), std::begin(results), [](size_t r) { return r / 2; });

	Logger csv;
	csv.log("GpuImpl1");
	csv.log("Size", "Stencils/Second");

	for (auto i = 0; i < sizes.size(); ++i) {
		csv.log(sizes[i], results[i]);
	}

	std::ofstream file("gpu-impl1.csv");
	csv.writeTo(file);
	file.close();
}

void test2() {
	std::fill(std::begin(results), std::end(results), 0);

	// jeweils 2x
	for (auto k = 0; k < 2; ++k) {
		// für alle Größen
		for (auto s = 0; s < sizes.size(); ++s) {
			auto size = sizes[s] + 2;

			float* input, *output;
			cudaMalloc(&input, sizeof(float) * size * size);
			cudaMalloc(&output, sizeof(float) * size * size);

			Matrix<float> data(size, size);

			cudaMemcpy(input, data.raw(), sizeof(float) * size * size, cudaMemcpyHostToDevice);

			dim3 blockSize { 32, 32, 1 };
			dim3 gridSize { (size - 2) / blockSize.x, (size - 2) / blockSize.y, 1 };

			GpuTimer timer;

			timer.start();

			for (auto i = 0; i < iterationsPerSize / 2; ++i) {
				fivePoint2<<<gridSize, blockSize>>>(input, output);
				fivePoint2<<<gridSize, blockSize>>>(output, input);
			}

			timer.stop();

			results[s] += stencilsPerSecond(size, size, timer.getDuration()) / iterationsPerSize;

			cudaMemcpy(data.raw(), input, sizeof(float) * size * size, cudaMemcpyDeviceToHost);

			cudaFree(output);
			cudaFree(input);
		}
	}

	std::transform(std::begin(results), std::end(results), std::begin(results), [](size_t r) { return r / 2; });

	Logger csv;
	csv.log("GpuImpl2");
	csv.log("Size", "Stencils/Second");

	for (auto i = 0; i < sizes.size(); ++i) {
		csv.log(sizes[i], results[i]);
	}

	std::ofstream file("gpu-impl2.csv");
	csv.writeTo(file);
	file.close();
}

int main() {
	test1();
	test2();
}

