#include <thrust/device_ptr.h>

__global__ void fivePoint2(thrust::device_ptr<float const> const input, thrust::device_ptr<float> const output, int width, int height) {
    auto x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    auto y = blockIdx.y * blockDim.y + threadIdx.y + 1;

    auto index = y * width + x;

    auto left = index - 1;
    auto right = index + 1;
    auto top = index - width;
    auto bottom = index + width;

    if (x >= width - 1 || y >= height - 1) {
        return;
    }

    output[index] = 1.0f / 5 * (input[top] + input[left] + input[index] + input[right] + input[bottom]);
}
