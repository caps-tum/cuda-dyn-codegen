#include <thrust/device_ptr.h>

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
