#include <iostream>
#include <iomanip>
#include <fstream>
#include <array>

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/device_new.h>
#include <thrust/device_delete.h>

#include <boost/program_options.hpp>

#include "Matrix.h"
#include "GpuTimer.h"
#include "Stencil.h"
#include "Logger.h"

namespace po = boost::program_options;

__global__ void fivePoint1(thrust::device_ptr<float const> const input, thrust::device_ptr<float> const output, size_t width, size_t height) {
	size_t x = blockIdx.x * blockDim.x + threadIdx.x;
	size_t y = blockIdx.y * blockDim.y + threadIdx.y;

	size_t index = y * width + x;

	size_t left = index - 1;
	size_t right = index + 1;
	size_t top = index - width;
	size_t bottom = index + width;

	if (x >= width || y >= height) {
		return;
	}

	if (x < 1 || y < 1 || x == width - 1 || y == height - 1) {
		output[index] = input[index];
	}
	else {
		output[index] = 1.0f / 5 * (input[top] + input[left] + input[index] + input[right] + input[bottom]);
	}
}

__global__ void fivePoint2(thrust::device_ptr<float const> const input, thrust::device_ptr<float> const output, size_t width, size_t height) {
	size_t x = blockIdx.x * blockDim.x + threadIdx.x + 1;
	size_t y = blockIdx.y * blockDim.y + threadIdx.y + 1;
	//size_t width = gridDim.x * blockDim.x + 2;
	//size_t height = gridDim.y * blockDim.y + 2;
	
	size_t index = y * width + x;

	size_t left = index - 1;
	size_t right = index + 1;
	size_t top = index - width;
	size_t bottom = index + width;

	if (x >= width - 1 || y >= height - 1) {
		return;
	}

	//output[index] = 1.0f / 5 * (input[left] + input[index] + input[right]+ input[top] + input[bottom]);
	// Wenn man die Zeile oben stattdessen so umsortiert, dass die zu addierenden Elemente in der Reihenfolge
	// aufsteigender Speicheradressen im Term auftauchen, gibt es einen großen Geschwindigkeitsschub.
	// 1380000 -> 1630000.
	output[index] = 1.0f / 5 * (input[top] + input[left] + input[index] + input[right] + input[bottom]);
}

__global__ void fivePoint3(thrust::device_ptr<float const> const input, thrust::device_ptr<float> const output, size_t global_width, size_t global_height) {
	size_t global_x = blockIdx.x * blockDim.x + threadIdx.x + 1;
	size_t global_y = blockIdx.y * blockDim.y + threadIdx.y + 1;

	size_t global_index = global_y * global_width + global_x;

	size_t global_left = global_index - 1;
	size_t global_right = global_index + 1;
	size_t global_top = global_index - global_width;
	size_t global_bottom = global_index + global_width;

	extern __shared__ float buffer[];

	size_t buffer_x = threadIdx.x + 1;
	size_t buffer_y = threadIdx.y + 1;
	
	size_t buffer_width = 18;
	size_t buffer_height = 18;

	int buffer_index = buffer_y * buffer_width + buffer_x;

	size_t buffer_left = buffer_index - 1;
	size_t buffer_right = buffer_index + 1;
	size_t buffer_top = buffer_index - buffer_width;
	size_t buffer_bottom = buffer_index + buffer_width;

	buffer[buffer_index] = input[global_index];

	if (threadIdx.x == 0) {
		buffer[buffer_left] = input[global_left];
	}

	if (threadIdx.x == buffer_width - 3) {
		buffer[buffer_right] = input[global_right];
	}

	if (threadIdx.y == 0) {
		buffer[buffer_top] = input[global_top];
	}

	if (threadIdx.y == buffer_height - 3) {
		buffer[buffer_bottom] = input[global_bottom];
	}
	__syncthreads();

	// Wenn man hier umsortiert, bringt das komischerweise gar nichts.
	// Allerdings ist er schneller, wenn man stattdessen die Variante mit dem 2d-Array benutzt, vielleicht kann er
	// dann besser optimieren.
	output[global_index] = 1.0f / 5 * (buffer[buffer_left] + buffer[buffer_index] + buffer[buffer_right] + buffer[buffer_top] + buffer[buffer_bottom]);
}


__global__ void fivePoint3_2(thrust::device_ptr<float const> const input, thrust::device_ptr<float> const output, size_t global_width, size_t global_height) {
	size_t global_x = blockIdx.x * blockDim.x + threadIdx.x + 1;
	size_t global_y = blockIdx.y * blockDim.y + threadIdx.y + 1;

	size_t global_index = global_y * global_width + global_x;

	size_t global_left = global_index - 1;
	size_t global_right = global_index + 1;
	size_t global_top = global_index - global_width;
	size_t global_bottom = global_index + global_width;

	__shared__ float buffer[18][18];

	size_t buffer_x = threadIdx.x + 1;
	size_t buffer_y = threadIdx.y + 1;

	size_t buffer_width = 18;
	size_t buffer_height = 18;

	int buffer_index = buffer_y * buffer_width + buffer_x;

	size_t buffer_left = buffer_index - 1;
	size_t buffer_right = buffer_index + 1;
	size_t buffer_top = buffer_index - buffer_width;
	size_t buffer_bottom = buffer_index + buffer_width;

	if (threadIdx.x == 0) {
	buffer[0][buffer_y] = input[global_left];
	}

	if (threadIdx.x == buffer_width - 3) {
	buffer[buffer_width - 1][buffer_y] = input[global_right];
	}

	if (threadIdx.y == 0) {
	buffer[buffer_x][0] = input[global_top];
	}

	if (threadIdx.y == buffer_height - 3) {
	buffer[buffer_x][buffer_height - 1] = input[global_bottom];
	}

	buffer[buffer_x][buffer_y] = input[global_index];

	__syncthreads();

	// Wenn man hier umsortiert, bringt das komischerweise gar nichts.
	// Allerdings ist er schneller, wenn man stattdessen die Variante mit dem 2d-Array benutzt, vielleicht kann er
	// dann besser optimieren.
	output[global_index] = 1.0f / 5 * (buffer[buffer_x][buffer_y - 1] + buffer[buffer_x - 1][buffer_y] + buffer[buffer_x][buffer_y] + buffer[buffer_x + 1][buffer_y] + buffer[buffer_x][buffer_y + 1]);
}

std::array<int, 5> sizes { 512 + 2, 1024 + 2, 2048 + 2, 4096 + 2, 8192 + 2 };
std::array<size_t, 5> results;

size_t iterationsPerSize = 60;

//std::array<int, 1> sizes { 32 + 2 };
//std::array<size_t, 1> results;

//size_t iterationsPerSize = 3;


void test1() {
	std::fill(std::begin(results), std::end(results), 0);

	// jeweils 2x
	for (auto k = 0; k < 2; ++k) {
		// für alle Größen
		for (auto s = 0; s < sizes.size(); ++s) {
			auto size = sizes[s];
			
			auto input = thrust::device_new<float>(size * size);
			auto output = thrust::device_new<float>(size * size);

			Matrix<float> data(size, size);

			thrust::copy_n(data.raw(), size * size, input);

			dim3 blockSize { 16, 16, 1 };
			dim3 gridSize { (size + blockSize.x - 1) / blockSize.x, (size + blockSize.y - 1) / blockSize.y, 1 };

			GpuTimer timer;

			timer.start();

			for (auto i = 0; i < iterationsPerSize; ++i) {
				fivePoint1<<<gridSize, blockSize>>>(input, output, size, size);
				thrust::swap(input, output);
			}

			timer.stop();

			results[s] += stencilsPerSecond(size, size, timer.getDuration()) / iterationsPerSize;

			thrust::copy_n(input, size * size, data.raw());

			if (size == 128) {
				std::ofstream file("data.txt");
				file << data;
				file.close();
			}
			
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
			auto size = sizes[s];

			auto input = thrust::device_new<float>(size * size);
			auto output = thrust::device_new<float>(size * size);

			Matrix<float> data(size, size);

			thrust::copy_n(data.raw(), size * size, input);
			thrust::copy_n(data.raw(), size * size, output);

			dim3 blockSize { 16, 16, 1 };
			dim3 gridSize { (size + blockSize.x - 1 - 2) / blockSize.x, (size + blockSize.y - 1 - 2) / blockSize.y, 1 };

			GpuTimer timer;

			timer.start();

			for (auto i = 0; i < iterationsPerSize; ++i) {
				fivePoint2<<<gridSize, blockSize>>>(input, output, size, size);
				thrust::swap(input, output);
			}

			timer.stop();

			results[s] += stencilsPerSecond(size, size, timer.getDuration()) / iterationsPerSize;

			thrust::copy_n(input, size * size, data.raw());

			/*if (size == 35 + 2) {
				std::ofstream file("data.txt");
				file << data;
				file.close();
			}*/

			thrust::device_delete(output, size * size);
			thrust::device_delete(input, size * size);
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

void test3() {
	std::fill(std::begin(results), std::end(results), 0);

	// jeweils 2x
	for (auto k = 0; k < 2; ++k) {
		// für alle Größen
		for (auto s = 0; s < sizes.size(); ++s) {
			auto size = sizes[s];

			auto input = thrust::device_new<float>(size * size);
			auto output = thrust::device_new<float>(size * size);

			Matrix<float> data(size, size);

			thrust::copy_n(data.raw(), size * size, input);
			thrust::copy_n(data.raw(), size * size, output);

			dim3 blockSize { 16, 16, 1 };
			dim3 gridSize { (size + blockSize.x - 1 - 2) / blockSize.x, (size + blockSize.y - 1 - 2) / blockSize.y, 1 };

			GpuTimer timer;

			timer.start();

			for (auto i = 0; i < iterationsPerSize; ++i) {
				//fivePoint3<<<gridSize, blockSize, sizeof(float) * 18 * 18>>>(input, output, size, size);
				fivePoint3_2 << <gridSize, blockSize>> >(input, output, size, size);
				thrust::swap(input, output);
			}

			timer.stop();

			results[s] += stencilsPerSecond(size, size, timer.getDuration()) / iterationsPerSize;

			thrust::copy_n(input, size * size, data.raw());

			if (size == 32 + 2) {
				std::ofstream file("data.txt");
				file << data;
				file.close();
			}

			thrust::device_delete(output, size * size);
			thrust::device_delete(input, size * size);
		}
	}

	std::transform(std::begin(results), std::end(results), std::begin(results), [](size_t r) { return r / 2; });

	Logger csv;
	csv.log("GpuImpl3");
	csv.log("Size", "Stencils/Second");

	for (auto i = 0; i < sizes.size(); ++i) {
		csv.log(sizes[i], results[i]);
	}

	std::ofstream file("gpu-impl3.csv");
	csv.writeTo(file);
	file.close();
}

int main(int argc, char* argv[]) {
	//test2();
	//test1();
	//test3();

	po::options_description desc("Allowed options");

	desc.add_options()
		("help", "produce help message")
		("kernel", po::value<int>(), "set the kernel version to use")
		;

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);


	std::cin.ignore();
}
