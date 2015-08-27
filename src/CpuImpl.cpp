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

std::array<size_t, 8> sizes { 128, 256, 512, 1024, 2048, 4096, 8192, 16384 }; // , 32768, 65536 };
std::array<size_t, 8> results;

size_t iterationsPerSize = 20;

int main() {
	// jeweils 2x
	for (auto k = 0; k < 2; ++k) {
		// für alle Größen
		for (auto s = 0; s < sizes.size(); ++s) {
			auto size = sizes[s];

			Matrix<float> a(size, size), b(size, size);

			FivePoint stencil;

			CpuTimer t;

			t.start();

			for (auto i = 0; i < iterationsPerSize / 2; ++i) {
				stencil(a, b, size, size);
				stencil(b, a, size, size);
			}

			t.stop();

			results[s] += stencilsPerSecond(size, size, t.getDuration()) / iterationsPerSize;
		}
	}

	transform(begin(results), end(results), begin(results), [](size_t r) { return r / 2; });

	Logger csv;
	csv.log("CpuImpl");
	csv.log("Size", "Stencils/Second");

	for (auto i = 0; i < sizes.size(); ++i) {
		csv.log(sizes[i], results[i]);
	}

	ofstream file("cpu-impl.csv");
	csv.writeTo(file);
	file.close();
}

