#pragma once

template <typename T>
class Matrix;

class Stencil {
public:
	using Operand = Matrix<float>;

	virtual Operand& operator()(Operand const& a, Operand& b, size_t width, size_t height) = 0;
};

