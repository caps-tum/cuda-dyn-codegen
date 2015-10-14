#include <fstream>
#include <iostream>
#include <memory>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <thrust/device_ptr.h>
#include <thrust/device_new.h>
#include <thrust/device_delete.h>

#include <boost/program_options.hpp>

#include "Cuda.h"
#include "CudaCall.h"
#include "GpuTimer.h"
#include "Matrix.h"
#include "Stencil.h"
#include "Logger.h"

#ifndef _WIN32
#include "make_unique.h"
#else
using std::make_unique;
#endif

CUfunction compile(std::string const& src, char const* name) {
    nvrtcProgram program;
    call(nvrtcCreateProgram(&program, src.data(), "source.cu", 0, nullptr, nullptr));

    char const* options[] = { "-std=c++11" };

    call(nvrtcCompileProgram(program, sizeof(options) / sizeof(options[1]), options));

#ifdef _DEBUG
    size_t logSize;
    nvrtcGetProgramLogSize(program, &logSize);

    auto log = make_unique<char[]>(logSize);

    nvrtcGetProgramLog(program, log.get());

    std::cout << "\n\nLOG:\n" << "====\n" << log.get();
#endif

    size_t ptxSize;
    nvrtcGetPTXSize(program, &ptxSize);

    auto ptx = make_unique<char[]>(ptxSize);

    call(nvrtcGetPTX(program, ptx.get()));

    call(nvrtcDestroyProgram(&program));

    CUmodule module;
    call(cuModuleLoadDataEx(&module, ptx.get(), 0, 0, 0));

    CUfunction kernel;
    call(cuModuleGetFunction(&kernel, module, name));

    return kernel;
}

std::string generateKernel(int matrixWidth, int sum, int stencilWidth, int stencilHeight, std::string const& stencilData) {
	std::ostringstream kernel;

	auto borderWidth = stencilWidth / 2;
	auto borderHeight = stencilHeight / 2;

	std::istringstream weights(stencilData);

	kernel
		<< "extern \"C\" __global__ void kernel(float const* input, float* output) { \n"
		<< "	auto x = blockIdx.x * blockDim.x + threadIdx.x + " << borderWidth << "; \n"
		<< "	auto y = blockIdx.y * blockDim.y + threadIdx.y + " << borderHeight << "; \n"
		<< "\n"
		<< "	auto index = y * " << matrixWidth << " + x; \n"
		<< "\n"
		;

	kernel
		<< "output[index] = 1.0f / " << sum << " * (";

	for (auto y = 0; y < stencilHeight; ++y) {
		for (auto x = 0; x < stencilWidth; ++x) {
			float w;
			weights >> w;

			// Sollte passen, weil beim Einlesen ein "0" ja hoffentlich in exakt 0.f umgewandelt wird.
			if (w != 0) {
				kernel
					<< w << " * input[index + " << (-stencilHeight / 2 + y) * matrixWidth << " + " << -stencilWidth / 2 + x << "] + \n";
			}
		}
	}

	kernel 
		<< " 0); \n"
		<< "} \n";

#ifdef _DEBUG
	std::cout << "Generated source: \n";
	std::cout << kernel.str() << std::endl;
#endif

	return kernel.str();
}


void runDyn(boost::program_options::variables_map const& vm) {
    auto width = vm["width"].as<int>();
    auto height = vm["height"].as<int>();
    auto numIterations = vm["numIterations"].as<int>();
    auto csvFile = vm["csv"].as<std::string>();

	auto sum = vm["sum"].as<float>();
	auto stencilWidth = vm["stencilWidth"].as<int>();
	auto stencilHeight = vm["stencilHeight"].as<int>();
	auto stencilData = vm["stencil"].as<std::string>();

	auto borderWidth = stencilWidth / 2;
	auto borderHeight = stencilHeight / 2;

    GpuTimer timer_all, timer_computation;

    timer_all.start();

	auto src = generateKernel(width, sum, stencilWidth, stencilHeight, stencilData);

    auto kernel = compile(src, "kernel");

    auto input = thrust::device_new<float>(width * height);
    auto output = thrust::device_new<float>(width * height);

    Matrix<float> data(width, height);
	data.addBorder(borderWidth);

    thrust::copy_n(data.raw(), width * height, input);
    thrust::copy_n(input, width * height, output);

    dim3 blockSize { 32, 8, 1 };
	dim3 gridSize { (width + blockSize.x - 1 - 2 * borderWidth) / blockSize.x, (height + blockSize.y - 1 - 2 * borderHeight) / blockSize.y, 1 };

    auto ip = input.get();
    auto op = output.get();

	void* args[] = { &ip, &op };

    timer_computation.start();

    for (auto i = 0; i < numIterations; ++i) {
        call(cuLaunchKernel(kernel, gridSize.x, gridSize.y, gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, nullptr, args, nullptr));

        std::swap(args[0], args[1]);
    }

    timer_computation.stop();

    thrust::copy_n(input, width * height, data.raw());

    thrust::device_delete(output, width * height);
    thrust::device_delete(input, width * height);

    timer_all.stop();

    Logger csv;
    csv.log("Dyn");
    csv.log("Width", "Height", "Stencils/Second (all)", "Stencils/Second (comput)");

	auto tAll = stencilsPerSecond(width - 2 * borderWidth, height - 2 * borderHeight, timer_all.getDuration() / numIterations);
	auto tComput = stencilsPerSecond(width - 2 * borderWidth, height - 2 * borderHeight, timer_computation.getDuration() / numIterations);

    csv.log(width, height, tAll, tComput);

    std::ofstream file(csvFile);
    csv.writeTo(file);
    file.close();

    if (vm.count("matrix")) {
        auto matrixFile = vm["matrix"].as<std::string>();

        std::ofstream m(matrixFile);
        m << data;
        m.close();
    }
}
