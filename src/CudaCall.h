#pragma once

void call(cudaError_t error) {
	if (error != cudaSuccess) {
		std::cerr << "Cuda Error: " << error << " - " << cudaGetErrorString(error) << "\n";
	}
}

