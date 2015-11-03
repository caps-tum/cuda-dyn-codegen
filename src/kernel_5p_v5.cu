#include <thrust/device_ptr.h>

__global__ void fivePoint5(thrust::device_ptr<float const> const input, thrust::device_ptr<float> const output, int global_width, int global_height) {
	auto global_x = blockIdx.x * blockDim.x + threadIdx.x + 1;
	auto global_y = blockIdx.y * blockDim.y + threadIdx.y + 1;

	auto global_index = global_y * global_width + global_x;

	auto global_left = global_index - 1;
	auto global_right = global_index + 1;
	auto global_top = global_index - global_width;
	auto global_bottom = global_index + global_width;
	/*
	__shared__ float buffer[34][10];
	//__shared__ float buffer[34 * 10];

	auto N = 32 * 8;
	auto S = 34 * 10;
	*/

	auto buffer_width = 34;
	auto buffer_height = 3;
	auto buffer_x = threadIdx.x + 1;
	auto buffer_y = threadIdx.y; // +1;
	

	/*
	Man hat 32 Threads für die Breite und drei für die Höhe.
	Man kann also drei Zeilen gleichzeitig verarbeiten.
	*/

	__shared__ float buffer[34][3];

	auto* b1 = &buffer[0][0];
	auto* b2 = &buffer[0][1];
	auto* b3 = &buffer[0][2];

	/*
	Zuerst müssen mal alle drei Zeilen mit Daten gefüllt werden.
	*/
	if (threadIdx.x == 0) {
		buffer[0][buffer_y] = input[global_left];
	}

	if (threadIdx.x == buffer_width - 3) {
		buffer[buffer_width - 1][buffer_y] = input[global_right];
	}

	buffer[buffer_x][buffer_y] = input[global_index];
	
	__syncthreads();

	/*
	Jetzt rechnet nur der mittlere Warp.

	An der Stelle könnte man auch noch eine vierte Zeile einführen, die *gleichzeitig* nachgeladen wird, also
	noch vor dem sync nach dem Rechnen. Man braucht tatsächlich auch keine 3 Warps, sondern nur 2.
	*/

	for (auto i = 1; i < global_height; ++i) {
		if (threadIdx.y == 1) {
			output[global_index + i * global_width] = 1.0f / 5 * (b1[buffer_x] + b2[buffer_x - 1] + b2[buffer_x] + b2[buffer_x + 1] + b3[buffer_x]);
		}

		__syncthreads();

		/*
		Jetzt werden die Zeilen "eins hochgeschoben" und die vierte unten nachgeladen.
		*/
		auto* bt = b1;
		b1 = b2;
		b2 = b3;
		b3 = bt;

		/*
		Also nach b3 laden.
		*/

		if (threadIdx.y == 0) {
			if (threadIdx.x == 0) {
				b3[0] = input[global_left];
			}

			if (threadIdx.x == buffer_width - 3) {
				b3[buffer_width - 1] = input[global_right];
			}

			b3[buffer_x] = input[global_index];
		}

		__syncthreads();

		/*
		Und wieder rechnet der mittlere Warp.
		*/

	}

}