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

#include "rt_kernel.h"

CUfunction compile(char const* src, char const* name) {
	nvrtcProgram program;
	call(nvrtcCreateProgram(&program, src, "source.cu", 0, nullptr, nullptr));

	char const* options[] = { "-std=c++11" };

	call(nvrtcCompileProgram(program, sizeof(options) / sizeof(options[1]), options));

#ifdef _DEBUG
	size_t logSize;
	nvrtcGetProgramLogSize(program, &logSize);

	auto log = std::make_unique<char[]>(logSize);

	nvrtcGetProgramLog(program, log.get());

	std::cout << "\n\nLOG:\n" << "====\n" << log.get();
#endif

	size_t ptxSize;
	nvrtcGetPTXSize(program, &ptxSize);

	auto ptx = std::make_unique<char[]>(ptxSize);

	call(nvrtcGetPTX(program, ptx.get()));

	call(nvrtcDestroyProgram(&program));

	CUmodule module;
	call(cuModuleLoadDataEx(&module, ptx.get(), 0, 0, 0));

	CUfunction kernel;
	call(cuModuleGetFunction(&kernel, module, name));

	return kernel;
}

void runDyn(boost::program_options::variables_map const& vm) {
	auto width = vm["width"].as<int>();
	auto height = vm["height"].as<int>();
	auto numIterations = vm["numIterations"].as<int>();
	auto csvFile = vm["csv"].as<std::string>();
	
	GpuTimer timer_all, timer_computation;

	timer_all.start();

	auto src = std::make_unique<char[]>(sizeof(ninePoint1_src) + 100);
	sprintf(src.get(), ninePoint1_src, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);

	auto kernel = compile(src.get(), "ninePoint1");

	auto result = 0;
	auto input = thrust::device_new<float>(width * height);
	auto output = thrust::device_new<float>(width * height);

	Matrix<float> data(width, height);
	data.addBorder(2);

	thrust::copy_n(data.raw(), width * height, input);
	thrust::copy_n(data.raw(), width * height, output);

	dim3 blockSize { 16, 16, 1 };
	dim3 gridSize { (width + blockSize.x - 1 - 4) / blockSize.x, (height + blockSize.y - 1 - 4) / blockSize.y, 1 };

	auto ip = input.get();
	auto op = output.get();

	void* args[] = { &ip, &op, &width, &height };

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

	auto tAll = stencilsPerSecond(width - 4, height - 4, timer_all.getDuration() / numIterations);
	auto tComput = stencilsPerSecond(width - 4, height - 4, timer_all.getDuration() / numIterations);

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
