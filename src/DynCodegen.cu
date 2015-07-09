#include <iostream>
#include <memory>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include "Cuda.h"
#include "GpuTimer.h"

char const* source =
"extern \"C\" __global__ void inc(float const* input, float* output) {"
"       output[threadIdx.y * blockDim.y + threadIdx.x] = input[threadIdx.y * blockDim.y + threadIdx.x] + 1;"
"}"
;

void call(nvrtcResult result) {
	if (result != NVRTC_SUCCESS) {
		std::cerr << "\nerror: " << result << " failed with error "
			<< nvrtcGetErrorString(result) << '\n';
		exit(1);
	}
}

void call(CUresult result) {
	if (result != CUDA_SUCCESS) {
		const char *msg;
		cuGetErrorName(result, &msg);
		std::cerr << "\nerror: " << result << " failed with error "
			<< msg << '\n';
		exit(1);
	}
}


Cuda cuda;

struct Kernel {
        CUfunction kernel;

        void launch(dim3 gridSize, dim3 blockSize, CUdeviceptr in, CUdeviceptr out) {
                void* args[] = { &in, &out };

                call(cuLaunchKernel(kernel, gridSize.x, gridSize.y, gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, nullptr, args, 0));
        }
};


int main() {
	GpuTimer timer;

	timer.start();

        nvrtcProgram prog;
        nvrtcCreateProgram(&prog, source, "source.cu", 0, nullptr, nullptr);

        nvrtcCompileProgram(prog, 0, nullptr);

        //size_t logSize;
        //nvrtcGetProgramLogSize(prog, &logSize);

        //char log[logSize];

        //nvrtcGetProgramLog(prog, log);

        //std::cout << "\n\nLOG:\n" << "====\n" << log << std::endl;

        size_t ptxSize;
        nvrtcGetPTXSize(prog, &ptxSize);

        char ptx[ptxSize];

        call(nvrtcGetPTX(prog, ptx));

        nvrtcDestroyProgram(&prog);

        CUmodule module;
        call(cuModuleLoadDataEx(&module, ptx, 0, 0, 0));

        CUfunction kernel;
        call(cuModuleGetFunction(&kernel, module, "inc"));

        CUdeviceptr data;
        cuMemAlloc(&data, 5 * 5 * sizeof(float));

        Kernel k = { kernel };
        k.launch(dim3(1, 1, 1), dim3(5, 5, 1), data, data);

        call(cuCtxSynchronize());

        float* dataH = new float[5 * 5];

        call(cuMemcpyDtoH(dataH, data, 5 * 5 * sizeof(float)));

	timer.stop();

        for (auto i = 0; i < 5 * 5; i++) {
                printf("%f; ", dataH[i]);
        }

	std::cout << "\n\n" << timer.getDuration().count() << " us\n";
}

