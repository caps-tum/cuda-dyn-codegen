#include <thrust/device_ptr.h>

__global__ void fivePoint4(thrust::device_ptr<float const> const input, thrust::device_ptr<float> const output, int global_width, int global_height) {
    auto global_x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    auto global_y = blockIdx.y * blockDim.y + threadIdx.y + 1;

    auto global_index = global_y * global_width + global_x;

    auto global_left = global_index - 1;
    auto global_right = global_index + 1;
    auto global_top = global_index - global_width;
    auto global_bottom = global_index + global_width;

    __shared__ float buffer[34][10];

    auto buffer_x = threadIdx.x + 1;
    auto buffer_y = threadIdx.y + 1;

    auto buffer_width = 34;
    auto buffer_height = 10;

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
