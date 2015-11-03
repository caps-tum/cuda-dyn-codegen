#include <iostream>
#include <iomanip>
#include <fstream>
#include <array>

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/device_new.h>
#include <thrust/device_delete.h>
#include <thrust/device_vector.h>

#include <boost/program_options.hpp>

#include "Matrix.h"
#include "GpuTimer.h"
#include "Stencil.h"
#include "Logger.h"

__global__ void fivePoint1(thrust::device_ptr<float const> const input, thrust::device_ptr<float> const output, int width, int height);
__global__ void fivePoint2(thrust::device_ptr<float const> const input, thrust::device_ptr<float> const output, int width, int height);
__global__ void fivePoint3(thrust::device_ptr<float const> const input, thrust::device_ptr<float> const output, int global_width, int global_height);
__global__ void fivePoint4(thrust::device_ptr<float const> const input, thrust::device_ptr<float> const output, int global_width, int global_height);
__global__ void fivePoint5(thrust::device_ptr<float const> const input, thrust::device_ptr<float> const output, int global_width, int global_height);
__global__ void ninePoint1(thrust::device_ptr<float const> const input, thrust::device_ptr<float> const output, int width);

void runGpu(boost::program_options::variables_map const& vm) {
    auto width = vm["width"].as<int>();
    auto height = vm["height"].as<int>();
    auto numIterations = vm["numIterations"].as<int>();
    auto csvFile = vm["csv"].as<std::string>();

    GpuTimer timer_all, timer_computation;

    timer_all.start();

    auto input = thrust::device_new<float>(width * height);
    auto output = thrust::device_new<float>(width * height);

    Matrix<float> data(width, height);


    switch (vm["kernel"].as<int>()) {
    case 1: {
        thrust::copy_n(data.raw(), width * height, input);

        dim3 blockSize { 16, 16, 1 };
        dim3 gridSize { (width + blockSize.x - 1) / blockSize.x, (width + blockSize.y - 1) / blockSize.y, 1 };

        timer_computation.start();

        for (auto i = 0; i < numIterations; ++i) {
            fivePoint1 << <gridSize, blockSize >> >(input, output, width, height);
            thrust::swap(input, output);
        }

        timer_computation.stop();

        thrust::copy_n(input, width * height, data.raw());

        thrust::device_delete(output, width * height);
        thrust::device_delete(input, width * height);

        timer_all.stop();

        Logger csv;
        csv.log("Gpu");
        csv.log("Width", "Height", "Stencils/Second (all)", "Stencils/Second (comput)");

        auto tAll = stencilsPerSecond(width - 2, height - 2, timer_all.getDuration() / numIterations);
        auto tComput = stencilsPerSecond(width - 2, height - 2, timer_computation.getDuration() / numIterations);

        csv.log(width, height, tAll, tComput);

        std::ofstream file(csvFile);
        csv.writeTo(file);
        file.close();

        if (vm.count("matrix")) {
            std::ofstream m(vm["matrix"].as<std::string>());
            m << data;
            m.close();
        }
    } break;
    case 2: {
        thrust::copy_n(data.raw(), width * height, input);
		thrust::copy_n(input, width * height, output);

        dim3 blockSize { 32, 8, 1 };
        dim3 gridSize { (width + blockSize.x - 1 - 2) / blockSize.x, (height + blockSize.y - 1 - 2) / blockSize.y, 1 };

        timer_computation.start();

		for (auto i = 0; i < numIterations; ++i) {
			fivePoint2 << <gridSize, blockSize >> >(input, output, width, height);
            thrust::swap(input, output);
        }

        timer_computation.stop();

        thrust::copy_n(input, width * height, data.raw());

        thrust::device_delete(output, width * height);
        thrust::device_delete(input, width * height);

        timer_all.stop();

        Logger csv;
        csv.log("Gpu");
        csv.log("Width", "Height", "Stencils/Second (all)", "Stencils/Second (comput)");

        auto tAll = stencilsPerSecond(width - 2, height - 2, timer_all.getDuration() / numIterations);
        auto tComput = stencilsPerSecond(width - 2, height - 2, timer_computation.getDuration() / numIterations);

        csv.log(width, height, tAll, tComput);

        std::ofstream file(csvFile);
        csv.writeTo(file);
        file.close();

        if (vm.count("matrix")) {
            std::ofstream m(vm["matrix"].as<std::string>());
            m << data;
            m.close();
        }
    } break;
    case 3: {
        thrust::copy_n(data.raw(), width * height, input);
        thrust::copy_n(input, width * height, output);

        dim3 blockSize { 32, 8, 1 };
        dim3 gridSize { (width + blockSize.x - 1 - 2) / blockSize.x, (height + blockSize.y - 1 - 2) / blockSize.y, 1 };

        timer_computation.start();

        for (auto i = 0; i < numIterations; ++i) {
            fivePoint3<<<gridSize, blockSize, sizeof(float) * 34 * 10>>>(input, output, width, height);
            thrust::swap(input, output);
        }

        timer_computation.stop();

        thrust::copy_n(input, width * height, data.raw());

        thrust::device_delete(output, width * height);
        thrust::device_delete(input, width * height);

        timer_all.stop();

        Logger csv;
        csv.log("Gpu");
        csv.log("Width", "Height", "Stencils/Second (all)", "Stencils/Second (comput)");

        auto tAll = stencilsPerSecond(width - 2, height - 2, timer_all.getDuration() / numIterations);
        auto tComput = stencilsPerSecond(width - 2, height - 2, timer_computation.getDuration() / numIterations);

        csv.log(width, height, tAll, tComput);

        std::ofstream file(csvFile);
        csv.writeTo(file);
        file.close();

        if (vm.count("matrix")) {
            std::ofstream m(vm["matrix"].as<std::string>());
            m << data;
            m.close();
        }
    } break;
    case 4: {
        thrust::copy_n(data.raw(), width * height, input);
        thrust::copy_n(input, width * height, output);

        dim3 blockSize { 32, 8, 1 };
        dim3 gridSize { (width + blockSize.x - 1 - 2) / blockSize.x, (height + blockSize.y - 1 - 2) / blockSize.y, 1 };

        timer_computation.start();

        for (auto i = 0; i < numIterations; ++i) {
            fivePoint4<<<gridSize, blockSize>>>(input, output, width, height);
            thrust::swap(input, output);
        }

        timer_computation.stop();

        thrust::copy_n(input, width * height, data.raw());

        thrust::device_delete(output, width * height);
        thrust::device_delete(input, width * height);

        timer_all.stop();

        Logger csv;
        csv.log("Gpu");
        csv.log("Width", "Height", "Stencils/Second (all)", "Stencils/Second (comput)");

        auto tAll = stencilsPerSecond(width - 2, height - 2, timer_all.getDuration() / numIterations);
        auto tComput = stencilsPerSecond(width - 2, height - 2, timer_computation.getDuration() / numIterations);

        csv.log(width, height, tAll, tComput);

        std::ofstream file(csvFile);
        csv.writeTo(file);
        file.close();

        if (vm.count("matrix")) {
            std::ofstream m(vm["matrix"].as<std::string>());
            m << data;
            m.close();
        }
    } break;
    case 5: {
		thrust::copy_n(data.raw(), width * height, input);
		thrust::copy_n(input, width * height, output);

		dim3 blockSize { 32, 3, 1 };
		dim3 gridSize { (width + blockSize.x - 1 - 2) / blockSize.x, (height + blockSize.y - 1 - 2) / blockSize.y, 1 };

		timer_computation.start();

		for (auto i = 0; i < numIterations; ++i) {
			fivePoint5 << <gridSize, blockSize >> >(input, output, width, height);
			thrust::swap(input, output);
		}

		timer_computation.stop();

		thrust::copy_n(input, width * height, data.raw());

		thrust::device_delete(output, width * height);
		thrust::device_delete(input, width * height);

		timer_all.stop();

		Logger csv;
		csv.log("Gpu");
		csv.log("Width", "Height", "Stencils/Second (all)", "Stencils/Second (comput)");

		auto tAll = stencilsPerSecond(width - 2, height - 2, timer_all.getDuration() / numIterations);
		auto tComput = stencilsPerSecond(width - 2, height - 2, timer_computation.getDuration() / numIterations);

		csv.log(width, height, tAll, tComput);

		std::ofstream file(csvFile);
		csv.writeTo(file);
		file.close();

		if (vm.count("matrix")) {
			std::ofstream m(vm["matrix"].as<std::string>());
			m << data;
			m.close();
		}
    } break;
    case 6: {
        data.addBorder(2);

        thrust::copy_n(data.raw(), width * height, input);
        thrust::copy_n(input, width * height, output);

        dim3 blockSize { 32, 8, 1 };
        dim3 gridSize { (width + blockSize.x - 1 - 4) / blockSize.x, (height + blockSize.y - 1 - 4) / blockSize.y, 1 };

        timer_computation.start();

        for (auto i = 0; i < numIterations; ++i) {
            ninePoint1 << <gridSize, blockSize >> >(input, output, width);
            thrust::swap(input, output);
        }

        timer_computation.stop();

        thrust::copy_n(input, width * height, data.raw());

        thrust::device_delete(output, width * height);
        thrust::device_delete(input, width * height);

        timer_all.stop();

        Logger csv;
        csv.log("Gpu");
        csv.log("Width", "Height", "Stencils/Second (all)", "Stencils/Second (comput)");

        auto tAll = stencilsPerSecond(width - 4, height - 4, timer_all.getDuration() / numIterations);
        auto tComput = stencilsPerSecond(width - 4, height - 4, timer_computation.getDuration() / numIterations);

        csv.log(width, height, tAll, tComput);

        std::ofstream file(csvFile);
        csv.writeTo(file);
        file.close();

        if (vm.count("matrix")) {
            std::ofstream m(vm["matrix"].as<std::string>());
            m << data;
            m.close();
        }
    } break;
    case 7: {

    } break;
    default:
        break;
    }
}
