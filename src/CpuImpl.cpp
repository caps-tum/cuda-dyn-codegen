#include <iostream>
#include <iomanip>
#include <algorithm>
#include <memory>
#include <fstream>
#include <sstream>
#include <map>
#include <functional>

#include "Stencil.h"
#include "Matrix.h"
#include "CpuTimer.h"
#include "Logger.h"

using namespace std;


struct FivePoint : public Stencil {
	virtual Operand& operator()(Operand const& a, Operand& b, size_t width, size_t height) override {
		for (auto y = 1; y < height - 1; ++y) {
			for (auto x = 1; x < width - 1; ++x) {
				b(x, y) = 1.0f / 5 * (a(x - 1, y) + a(x, y - 1) + a(x, y) + a(x + 1, y) + a(x, y + 1));
			}
		}

		return b;
	}
};


size_t const width = 1024, height = 1024;

int main() {
	Matrix<float> a(width, height), b(width, height);
	
	FivePoint stencil;
	
	CpuTimer t;

	t.start();

	for (auto i = 0; i < 20; ++i) {
		stencil(a, b, width, height);
		stencil(b, a, width, height);
	}

	t.stop();

	cout << "Duration, 40 iterations: " << to_string(t.getDuration()) << "\n";

}

