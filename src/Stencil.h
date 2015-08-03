#pragma once

#include <chrono>

template <typename T>
class Matrix;

class Stencil {
public:
	using Operand = Matrix<float>;

	virtual Operand& operator()(Operand const& a, Operand& b, size_t width, size_t height) = 0;
};

size_t stencilsPerSecond(size_t width, size_t height, std::chrono::microseconds duration) {
	using fpSeconds = std::chrono::duration<double, std::chrono::seconds::period>;

	return (width - 2) * (height - 2) / std::chrono::duration_cast<fpSeconds>(duration).count();
}
