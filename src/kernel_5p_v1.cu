#include <thrust/device_ptr.h>

__global__ void fivePoint1(thrust::device_ptr<float const> const input, thrust::device_ptr<float> const output, int width, int height) {
    auto x = blockIdx.x * blockDim.x + threadIdx.x;
    auto y = blockIdx.y * blockDim.y + threadIdx.y;

    auto index = y * width + x;

    auto left = index - 1;
    auto right = index + 1;
    auto top = index - width;
    auto bottom = index + width;

    if (x >= width || y >= height) {
        return;
    }

    if (x < 1 || y < 1 || x == width - 1 || y == height - 1) {
        output[index] = input[index];
    }
    else {
        output[index] = 1.0f / 5 * (input[top] + input[left] + input[index] + input[right] + input[bottom]);
    }
}
