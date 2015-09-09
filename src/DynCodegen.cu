#include <iostream>
#include <memory>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include "Cuda.h"
#include "CudaCall.h"
#include "GpuTimer.h"

#include <fstream>

#include <thrust/device_ptr.h>
#include <thrust/device_new.h>
#include <thrust/device_delete.h>

#include "Matrix.h"
#include "Stencil.h"
#include "Logger.h"
#include "Test.h"

#include "rt_kernel.h"


auto iterationsPerSize = 50;

int main() {

	auto src = std::make_unique<char[]>(sizeof(ninePoint1_src) + 100);
	sprintf(src.get(), ninePoint1_src, 1, 1, 1, 1, 1, 1, 1, 1, 1);

	GpuTimer timer;

	timer.start();
	
    nvrtcProgram prog;
	nvrtcCreateProgram(&prog, src.get(), "source.cu", 0, nullptr, nullptr);

    nvrtcCompileProgram(prog, 0, nullptr);

    size_t logSize;
    nvrtcGetProgramLogSize(prog, &logSize);

	auto log = std::make_unique<char[]>(logSize);

    nvrtcGetProgramLog(prog, log.get());

    std::cout << "\n\nLOG:\n" << "====\n" << log.get() << std::endl;

    size_t ptxSize;
    nvrtcGetPTXSize(prog, &ptxSize);

	auto ptx = std::make_unique<char[]>(ptxSize);

    call(nvrtcGetPTX(prog, ptx.get()));

    nvrtcDestroyProgram(&prog);

    CUmodule module;
    call(cuModuleLoadDataEx(&module, ptx.get(), 0, 0, 0));

    CUfunction kernel;
    call(cuModuleGetFunction(&kernel, module, "ninePoint1"));

	auto result = 0;
	int size = 32 + 4;
	auto input = thrust::device_new<float>(size * size);
	auto output = thrust::device_new<float>(size * size);
		
	Matrix<float> data(size, size);

	thrust::copy_n(data.raw(), size * size, input);
	thrust::copy_n(data.raw(), size * size, output);

	dim3 blockSize { 16, 16, 1 };
	dim3 gridSize { (size + blockSize.x - 1 - 4) / blockSize.x, (size + blockSize.y - 1 - 4) / blockSize.y, 1 };

	//GpuTimer timer;

	//timer.start();

	/*thrust::device_vector<int> weights = std::vector<int> {
		0, 0, 1, 0, 0,
		0, 0, 1, 0, 0,
		1, 1, 1, 1, 1,
		0, 0, 1, 0, 0,
		0, 0, 1, 0, 0
		};*/

	auto ip = input.get();
	auto op = output.get();
			
	// Wieso geht das, wenn size ein int ist, aber nicht, wenn es ein size_t ist?
	void* args[] = { &ip, &op, &size, &size };

	for (auto i = 0; i < iterationsPerSize; ++i) {
		call(cuLaunchKernel(kernel, gridSize.x, gridSize.y, gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, nullptr, args, nullptr));

		std::swap(args[0], args[1]);
	}

	timer.stop();

	result += stencilsPerSecond(size, size, timer.getDuration()) / iterationsPerSize;

	thrust::copy_n(input, size * size, data.raw());

	std::cout << result;

	if (size == 32 + 4) {
		std::ofstream file("data.txt");
		file << data;
		file.close();
	}

	thrust::device_delete(output, size * size);
	thrust::device_delete(input, size * size);
}
