#include <iostream>
#include <iomanip>
#include <fstream>
#include <array>

#include <cuda.h>
#include <cuda_runtime.h>

#include "Matrix.h"
#include "GpuTimer.h"
#include "Stencil.h"
#include "Logger.h"

using namespace std;

__global__ void fivePoint1(float const* input, float* output) {
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
	size_t x = blockIdx.x * blockDim.x + threadIdx.x;
	size_t y = blockIdx.y * blockDim.y + threadIdx.y;
	size_t width = gridDim.x * blockDim.x;
	size_t height = gridDim.y * blockDim.y;
	size_t index = (y + 1) * width + x + 1;

	size_t left = index - 1;
	size_t right = index + 1;
	size_t top = index - width;
	size_t bottom = index + width;

	//if (x < 1 || y < 1 || x == width - 1 || y == height - 1) {
	//	output[index] = input[index];
	//}
	//else {
		output[index] = 1.0f / 5 * (input[left] + input[index] + input[right] + input[top] + input[bottom]);
	//}
}

/*__global__ void fivePoint_shared(float const* input, float* output) {
	size_t x = blockIdx.x * blockDim.x + threadIdx.x;
	size_t y = blockIdx.y * blockDim.y + threadIdx.y;
	size_t width = gridDim.x * blockDim.x;
	size_t height = gridDim.y * blockDim.y;
	size_t index = y * width + x;

	__shared__ float sharedbuffer[32][32];

	if (threadIdx.x < warpSize) {
		for (auto i = threadIdx; i < 32; ++i) {
			sharedbuffer[i][threadIdx] = input[index];
		}
	}

	// Die Schleife ist doch falsch?
	// Ganz abgesehen davon brauchen wir ja im shared buffer auch noch diese
	// holo region für den stencil.
	// Wenn man einen Warp zum Laden einsetzen will, wer lädt denn dann die holo region?
	// Der shared memory wird sich von allen threads im selben block geteilt.
	// 

	size_t localIndex = threadIdx.y * blockDim.y + threadIdx.x;


	__syncthreads();

	size_t left = localIndex - 1;
	size_t right = localIndex + 1;
	size_t top = localIndex - blockDim.y;
	size_t bottom = localIndex + blockDim.y;

	if (x < 1 || y < 1 || x == width - 1 || y == height - 1) {
		//output[index] = input[index];
		output[index] = sharedbuffer[localIndex];
	}
	else {
		output[index] = 1.0f / 5 * (sharedbuffer[left] + sharedbuffer[localIndex] + sharedbuffer[right] + sharedbuffer[top] + sharedbuffer[bottom]);
		//output[index] = 1.0f / 5 * (input[left] + input[index] + input[right] + input[top] + input[bottom]);
	}
}*/

/*__global__ void fivePoint(float const* input, float* output) {
	size_t x = blockIdx.x * blockDim.x + threadIdx.x;
	size_t y = blockIdx.y * blockDim.y + threadIdx.y;
	size_t width = gridDim.x * blockDim.x;
	size_t height = gridDim.y * blockDim.y;
	size_t index = y * width + x;

	size_t left = index - 1;
	size_t right = index + 1;
	size_t top = index - width;
	size_t bottom = index + width;
*/
	/*if (x < 1 || y < 1 || x == width - 1 || y == height - 1) {
		output[index] = input[index];
	}
	else {
		output[index] = 1.0f / 5 * (input[left] + input[index] + input[right] + input[top] + input[bottom]);
	}*/

	// zefix, was heißt warp divergence? dass mehrere threads innerhalb einer warps, also innerhalb einer 32er-gruppe unterschiedliche
	// pfade durchlaufen.
	// wie soll man das verhindern können in den randbereichen?
	// man muss quasi die arbeit anders verteilen, und kein 1:1-mapping von x und y auf die array-indizes durchführen.

//}

std::array<int, 5> sizes { 128, 256, 512, 1024, 2048 };
std::array<size_t, 5> results;

size_t iterationsPerSize = 60;

int main() {
	// jeweils 2x
	for (auto k = 0; k < 2; ++k) {
		// für alle Größen
		for (auto s = 0; s < sizes.size(); ++s) {
			auto size = sizes[s];

			float* input, * output;
			cudaMalloc(&input, sizeof(float) * size * size);
			cudaMalloc(&output, sizeof(float) * size * size);

			Matrix<float> data(size, size);

			cudaMemcpy(input, data.raw(), sizeof(float) * size * size, cudaMemcpyHostToDevice);

			dim3 blockSize { 32, 32, 1 };
			dim3 gridSize { size / blockSize.x, size / blockSize.y, 1 };

			GpuTimer timer;

			timer.start();

			for (auto i = 0; i < iterationsPerSize / 2; ++i) {
				fivePoint1<<<gridSize, blockSize>>>(input, output);
				fivePoint1<<<gridSize, blockSize>>>(output, input);
			}

			timer.stop();

			results[s] += stencilsPerSecond(size, size, timer.getDuration()) / iterationsPerSize;

			cudaMemcpy(data.raw(), input, sizeof(float) * size * size, cudaMemcpyDeviceToHost);

			cudaFree(output);
			cudaFree(input);
		}
	}

	transform(begin(results), end(results), begin(results), [](size_t r) { return r / 2; });

	Logger csv;
	csv.log("GpuImpl");
	csv.log("Size", "Stencils/Second");

	for (auto i = 0; i < sizes.size(); ++i) {
		csv.log(sizes[i], results[i]);
	}

	ofstream file("gpu-impl.csv");
	csv.writeTo(file);
	file.close();

	//std::cout << data;

	//std::ofstream file("file.txt");
	//file << data;
	//file.close();

	//std::cout << "\nDauer: " << timer.getDuration().count() << " us" << std::endl;

}

