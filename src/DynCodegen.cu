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
	nvrtcProgram prog;
	nvrtcCreateProgram(&prog, src, "source.cu", 0, nullptr, nullptr);

	nvrtcCompileProgram(prog, 0, nullptr);

	size_t logSize;
	nvrtcGetProgramLogSize(prog, &logSize);

	auto log = std::make_unique<char[]>(logSize);

	nvrtcGetProgramLog(prog, log.get());

	std::cout << "\n\nLOG:\n" << "====\n" << log.get();

	size_t ptxSize;
	nvrtcGetPTXSize(prog, &ptxSize);

	auto ptx = std::make_unique<char[]>(ptxSize);

	call(nvrtcGetPTX(prog, ptx.get()));

	nvrtcDestroyProgram(&prog);

	CUmodule module;
	call(cuModuleLoadDataEx(&module, ptx.get(), 0, 0, 0));

	CUfunction kernel;
	call(cuModuleGetFunction(&kernel, module, name));

	return kernel;
}

static GpuTimer::duration run(CUfunction kernel, int width, int height, int numIterations) {
	GpuTimer timer_computation;

	auto result = 0;
	auto input = thrust::device_new<float>(width * height);
	auto output = thrust::device_new<float>(width * height);

	Matrix<float> data(width, height);

	thrust::copy_n(data.raw(), width * height, input);
	thrust::copy_n(data.raw(), width * height, output);

	dim3 blockSize { 16, 16, 1 };
	dim3 gridSize { (width + blockSize.x - 1 - 4) / blockSize.x, (height + blockSize.y - 1 - 4) / blockSize.y, 1 };

	auto ip = input.get();
	auto op = output.get();

	// Wieso geht das, wenn size ein int ist, aber nicht, wenn es ein size_t ist?
	void* args[] = { &ip, &op, &width, &height };

	timer_computation.start();

	for (auto i = 0; i < numIterations; ++i) {
		call(cuLaunchKernel(kernel, gridSize.x, gridSize.y, gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, nullptr, args, nullptr));

		std::swap(args[0], args[1]);
	}

	timer_computation.stop();

	thrust::copy_n(input, width * height, data.raw());

	/*if (size == 32 + 4) {
	std::ofstream file("data.txt");
	file << data;
	file.close();
	}*/

	thrust::device_delete(output, width * height);
	thrust::device_delete(input, width * height);

	return timer_computation.getDuration();
}

void runDyn(boost::program_options::variables_map const& vm) {
	auto width = vm["width"].as<int>();
	auto height = vm["height"].as<int>();
	auto numIterations = vm["numIterations"].as<int>();
	auto output = vm["output"].as<std::string>();

	GpuTimer timer_all;

	timer_all.start();

	auto src = std::make_unique<char[]>(sizeof(ninePoint1_src) + 100);
	sprintf(src.get(), ninePoint1_src, 1, 1, 1, 1, 1, 1, 1, 1, 1);

	auto kernel = compile(src.get(), "ninePoint1");

	auto duration_computation = run(kernel, width, height, numIterations);

	timer_all.stop();

	Logger csv;
	csv.log("Dyn");
	csv.log("Width", "Height", "Stencils/Seconds (total)", "Stencils/Second (comput)");

	csv.log(width, height, stencilsPerSecond(width, height, timer_all.getDuration()) / numIterations, stencilsPerSecond(width, height, duration_computation) / numIterations);

	std::ofstream file(output);
	csv.writeTo(file);
	file.close();
}
