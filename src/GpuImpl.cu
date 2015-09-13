#include <iostream>
#include <iomanip>
#include <fstream>
#include <array>

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/device_new.h>
#include <thrust/device_delete.h>
#include <thrust/device_vector.h>

#include <boost/program_options.hpp>

#include "Matrix.h"
#include "GpuTimer.h"
#include "Stencil.h"
#include "Logger.h"

__global__ void fivePoint1(thrust::device_ptr<float const> const input, thrust::device_ptr<float> const output, int width, int height) {
	auto x = blockIdx.x * blockDim.x + threadIdx.x;
	auto y = blockIdx.y * blockDim.y + threadIdx.y;

	auto index = y * width + x;

	auto left = index - 1;
	auto right = index + 1;
	auto top = index - width;
	auto bottom = index + width;

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

__global__ void fivePoint2(thrust::device_ptr<float const> const input, thrust::device_ptr<float> const output, int width, int height) {
	auto x = blockIdx.x * blockDim.x + threadIdx.x + 1;
	auto y = blockIdx.y * blockDim.y + threadIdx.y + 1;

	auto index = y * width + x;

	auto left = index - 1;
	auto right = index + 1;
	auto top = index - width;
	auto bottom = index + width;

	if (x >= width - 1 || y >= height - 1) {
		return;
	}

	output[index] = 1.0f / 5 * (input[top] + input[left] + input[index] + input[right] + input[bottom]);
}

__global__ void fivePoint3(thrust::device_ptr<float const> const input, thrust::device_ptr<float> const output, int global_width, int global_height) {
	auto global_x = blockIdx.x * blockDim.x + threadIdx.x + 1;
	auto global_y = blockIdx.y * blockDim.y + threadIdx.y + 1;

	auto global_index = global_y * global_width + global_x;

	auto global_left = global_index - 1;
	auto global_right = global_index + 1;
	auto global_top = global_index - global_width;
	auto global_bottom = global_index + global_width;

	extern __shared__ float buffer[];

	auto buffer_x = threadIdx.x + 1;
	auto buffer_y = threadIdx.y + 1;
	
	auto buffer_width = 18;
	auto buffer_height = 18;

	int buffer_index = buffer_y * buffer_width + buffer_x;

	auto buffer_left = buffer_index - 1;
	auto buffer_right = buffer_index + 1;
	auto buffer_top = buffer_index - buffer_width;
	auto buffer_bottom = buffer_index + buffer_width;

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

	output[global_index] = 1.0f / 5 * (buffer[buffer_top] + buffer[buffer_left] + buffer[buffer_index] + buffer[buffer_right] + buffer[buffer_bottom]);
}


__global__ void fivePoint4(thrust::device_ptr<float const> const input, thrust::device_ptr<float> const output, int global_width, int global_height) {
	auto global_x = blockIdx.x * blockDim.x + threadIdx.x + 1;
	auto global_y = blockIdx.y * blockDim.y + threadIdx.y + 1;

	auto global_index = global_y * global_width + global_x;

	auto global_left = global_index - 1;
	auto global_right = global_index + 1;
	auto global_top = global_index - global_width;
	auto global_bottom = global_index + global_width;

	__shared__ float buffer[18][18];

	auto buffer_x = threadIdx.x + 1;
	auto buffer_y = threadIdx.y + 1;

	auto buffer_width = 18;
	auto buffer_height = 18;

	auto buffer_index = buffer_y * buffer_width + buffer_x;

	auto buffer_left = buffer_index - 1;
	auto buffer_right = buffer_index + 1;
	auto buffer_top = buffer_index - buffer_width;
	auto buffer_bottom = buffer_index + buffer_width;

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

	output[global_index] = 1.0f / 5 * (buffer[buffer_x][buffer_y - 1] + buffer[buffer_x - 1][buffer_y] + buffer[buffer_x][buffer_y] + buffer[buffer_x + 1][buffer_y] + buffer[buffer_x][buffer_y + 1]);
}

__global__ void ninePoint1(thrust::device_ptr<float const> const input, thrust::device_ptr<float> const output, int width, int height, thrust::device_ptr<int const> const weights) {
	auto x = blockIdx.x * blockDim.x + threadIdx.x + 2;
	auto y = blockIdx.y * blockDim.y + threadIdx.y + 2;

	auto index = y * width + x;

	if (x >= width - 2 || y >= height - 2) {
		return;
	}

	auto sum =
		weights[0 * 5 + 2] +
		weights[1 * 5 + 2] +
		weights[2 * 5 + 0] +
		weights[2 * 5 + 1] +
		weights[2 * 5 + 2] +
		weights[2 * 5 + 3] +
		weights[2 * 5 + 4] +
		weights[3 * 5 + 2] +
		weights[4 * 5 + 2]
	;

	output[index] = 1.0f / sum * (
		weights[0 * 5 + 2] * input[index - width - width] +
		weights[1 * 5 + 2] * input[index - width] +
		weights[2 * 5 + 0] * input[index - 1 - 1] +
		weights[2 * 5 + 1] * input[index - 1] +
		weights[2 * 5 + 2] * input[index] +
		weights[2 * 5 + 3] * input[index + 1] +
		weights[2 * 5 + 4] * input[index + 1 + 1] +
		weights[3 * 5 + 2] * input[index + width] +
		weights[4 * 5 + 2] * input[index + width + width]
	);
}


void runGpu(boost::program_options::variables_map const& vm) {
	auto width = vm["width"].as<int>();
	auto height = vm["height"].as<int>();
	auto numIterations = vm["numIterations"].as<int>();
	auto csvFile = vm["csv"].as<std::string>();

	GpuTimer timer_all, timer_computation;

	timer_all.start();

	auto input = thrust::device_new<float>(width * height);
	auto output = thrust::device_new<float>(width * height);

	Matrix<float> data(width, height);


	switch (vm["kernel"].as<int>()) {
	case 1: {
		timer_all.start();

		thrust::copy_n(data.raw(), width * height, input);

		dim3 blockSize { 16, 16, 1 };
		dim3 gridSize { (width + blockSize.x - 1) / blockSize.x, (width + blockSize.y - 1) / blockSize.y, 1 };

		timer_computation.start();

		for (auto i = 0; i < numIterations; ++i) {
			fivePoint1 << <gridSize, blockSize >> >(input, output, width, height);
			thrust::swap(input, output);
		}

		timer_computation.stop();

		thrust::copy_n(input, width * height, data.raw());

		thrust::device_delete(output, width * height);
		thrust::device_delete(input, width * height);

		timer_all.stop();

		Logger csv;
		csv.log("Gpu");
		csv.log("Width", "Height", "Stencils/Second (all)", "Stencils/Second (comput)");

		auto tAll = stencilsPerSecond(width - 2, height - 2, timer_all.getDuration() / numIterations);
		auto tComput = stencilsPerSecond(width - 2, height - 2, timer_all.getDuration() / numIterations);

		csv.log(width, height, tAll, tComput);

		std::ofstream file(csvFile);
		csv.writeTo(file);
		file.close();

		if (vm.count("matrix")) {
			std::ofstream m(vm["matrix"].as<std::string>());
			m << data;
			m.close();
		}
	} break;
	case 2: {
		timer_all.start();

		thrust::copy_n(data.raw(), width * height, input);
		thrust::copy_n(data.raw(), width * height, output);

		dim3 blockSize { 16, 16, 1 };
		dim3 gridSize { (width + blockSize.x - 1 - 2) / blockSize.x, (height + blockSize.y - 1 - 2) / blockSize.y, 1 };

		timer_computation.start();

		for (auto i = 0; i < numIterations; ++i) {
			fivePoint2 << <gridSize, blockSize >> >(input, output, width, height);
			thrust::swap(input, output);
		}

		timer_computation.stop();

		thrust::copy_n(input, width * height, data.raw());

		thrust::device_delete(output, width * height);
		thrust::device_delete(input, width * height);

		timer_all.stop();

		Logger csv;
		csv.log("Gpu");
		csv.log("Width", "Height", "Stencils/Second (all)", "Stencils/Second (comput)");

		auto tAll = stencilsPerSecond(width - 2, height - 2, timer_all.getDuration() / numIterations);
		auto tComput = stencilsPerSecond(width - 2, height - 2, timer_all.getDuration() / numIterations);

		csv.log(width, height, tAll, tComput);

		std::ofstream file(csvFile);
		csv.writeTo(file);
		file.close();

		if (vm.count("matrix")) {
			std::ofstream m(vm["matrix"].as<std::string>());
			m << data;
			m.close();
		}
	} break;
	case 3: {
		timer_all.start();

		thrust::copy_n(data.raw(), width * height, input);
		thrust::copy_n(data.raw(), width * height, output);

		dim3 blockSize { 16, 16, 1 };
		dim3 gridSize { (width + blockSize.x - 1 - 2) / blockSize.x, (height + blockSize.y - 1 - 2) / blockSize.y, 1 };
		
		timer_computation.start();

		for (auto i = 0; i < numIterations; ++i) {
			fivePoint3<<<gridSize, blockSize, sizeof(float) * 18 * 18>>>(input, output, width, height);
			thrust::swap(input, output);
		}

		timer_computation.stop();

		thrust::copy_n(input, width * height, data.raw());

		thrust::device_delete(output, width * height);
		thrust::device_delete(input, width * height);

		timer_all.stop();

		Logger csv;
		csv.log("Gpu");
		csv.log("Width", "Height", "Stencils/Second (all)", "Stencils/Second (comput)");

		auto tAll = stencilsPerSecond(width - 2, height - 2, timer_all.getDuration() / numIterations);
		auto tComput = stencilsPerSecond(width - 2, height - 2, timer_all.getDuration() / numIterations);

		csv.log(width, height, tAll, tComput);

		std::ofstream file(csvFile);
		csv.writeTo(file);
		file.close();

		if (vm.count("matrix")) {
			std::ofstream m(vm["matrix"].as<std::string>());
			m << data;
			m.close();
		}
	} break;
	case 4: {
		timer_all.start();

		thrust::copy_n(data.raw(), width * height, input);
		thrust::copy_n(data.raw(), width * height, output);

		dim3 blockSize { 16, 16, 1 };
		dim3 gridSize { (width + blockSize.x - 1 - 2) / blockSize.x, (height + blockSize.y - 1 - 2) / blockSize.y, 1 };

		timer_computation.start();

		for (auto i = 0; i < numIterations; ++i) {
			fivePoint4<<<gridSize, blockSize>>>(input, output, width, height);
			thrust::swap(input, output);
		}

		timer_computation.stop();

		thrust::copy_n(input, width * height, data.raw());

		thrust::device_delete(output, width * height);
		thrust::device_delete(input, width * height);

		timer_all.stop();

		Logger csv;
		csv.log("Gpu");
		csv.log("Width", "Height", "Stencils/Second (all)", "Stencils/Second (comput)");

		auto tAll = stencilsPerSecond(width - 2, height - 2, timer_all.getDuration() / numIterations);
		auto tComput = stencilsPerSecond(width - 2, height - 2, timer_all.getDuration() / numIterations);

		csv.log(width, height, tAll, tComput);

		std::ofstream file(csvFile);
		csv.writeTo(file);
		file.close();

		if (vm.count("matrix")) {
			std::ofstream m(vm["matrix"].as<std::string>());
			m << data;
			m.close();
		}
	} break;
	case 5: {

	} break;
	case 6: {		
		data.addBorder(2);

		thrust::copy_n(data.raw(), width * height, input);
		thrust::copy_n(data.raw(), width * height, output);

		dim3 blockSize { 16, 16, 1 };
		dim3 gridSize { (width + blockSize.x - 1 - 4) / blockSize.x, (height + blockSize.y - 1 - 4) / blockSize.y, 1 };

		timer_computation.start();

		thrust::device_vector<int> weights = std::vector<int> {
			 0, 0, 1, 0, 0,
			 0, 0, 1, 0, 0,
			 1, 1, 1, 1, 1,
			 0, 0, 1, 0, 0,
			 0, 0, 1, 0, 0
		};	
		

		for (auto i = 0; i < numIterations; ++i) {
			ninePoint1 << <gridSize, blockSize >> >(input, output, width, height, weights.data());
			thrust::swap(input, output);
		}

		timer_computation.stop();

		thrust::copy_n(input, width * height, data.raw());
		
		thrust::device_delete(output, width * height);
		thrust::device_delete(input, width * height);

		timer_all.stop();

		Logger csv;
		csv.log("Gpu");
		csv.log("Width", "Height", "Stencils/Second (all)", "Stencils/Second (comput)");

		auto tAll = stencilsPerSecond(width - 4, height - 4, timer_all.getDuration() / numIterations);
		auto tComput = stencilsPerSecond(width - 4, height - 4, timer_all.getDuration() / numIterations);

		csv.log(width, height, tAll, tComput);
		
		std::ofstream file(csvFile);
		csv.writeTo(file);
		file.close();

		if (vm.count("matrix")) {
			std::ofstream m(vm["matrix"].as<std::string>());
			m << data;
			m.close();
		}
	} break;
	case 7: {

	} break;
	default:
		break;
	}
}


