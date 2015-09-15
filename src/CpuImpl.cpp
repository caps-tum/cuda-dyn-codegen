#include <assert.h>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <memory>
#include <fstream>
#include <sstream>
#include <map>
#include <array>
#include <functional>

#include <boost/program_options.hpp>

#include "Stencil.h"
#include "Matrix.h"
#include "CpuTimer.h"
#include "Logger.h"

using namespace std;

struct FivePoint {
    using Operand = Matrix<float>;

    Operand& operator()(Operand const& a, Operand& b, size_t width, size_t height) {
        for (auto y = 1; y < height - 1; ++y) {
            for (auto x = 1; x < width - 1; ++x) {
                b(x, y) = 1.0f / 5 * (a(x - 1, y) + a(x, y - 1) + a(x, y) + a(x + 1, y) + a(x, y + 1));
            }
        }

        return b;
    }
};

void runCpu(boost::program_options::variables_map const& vm) {
    auto width = vm["width"].as<int>();
    auto height = vm["height"].as<int>();
    auto numIterations = vm["numIterations"].as<int>();
    auto csvFile = vm["csv"].as<std::string>();

    CpuTimer timer_all, timer_computation;

    timer_all.start();

    Matrix<float> a(width, height), b(width, height);

    FivePoint stencil;

    timer_computation.start();

    for (auto i = 0; i < numIterations; ++i) {
        stencil(a, b, width, height);
        swap(a, b);
    }

    timer_computation.stop();
    timer_all.stop();

    Logger csv;
    csv.log("CpuImpl");
    csv.log("Width", "Height", "NumIterations", "Stencils/Second (all)", "Stencils/Second (comput)");

    auto tAll = stencilsPerSecond(width - 2, height - 2, timer_all.getDuration() / numIterations);
    auto tComput = stencilsPerSecond(width - 2, height - 2, timer_computation.getDuration() / numIterations);

    csv.log(width, height, numIterations, tAll, tComput);

    ofstream file(csvFile);
    csv.writeTo(file);
    file.close();

    if (vm.count("matrix")) {
        auto matrixFile = vm["matrix"].as<std::string>();

        ofstream m(matrixFile);
        m << a;
        m.close();
    }
}
