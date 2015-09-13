#pragma once

#include <algorithm>
#include <iosfwd>
#include <iomanip>
#include <memory>

template <typename T>
class Matrix {
public:
    using value_type = T;
    using pointer = value_type*;
    using reference = value_type&;
    using const_reference = value_type const&;

    using this_type = Matrix<T>;

    Matrix(size_t w, size_t h)
        : width { w }
        , height { h }
        , matrix { new value_type[width * height] }
    {
        std::fill(begin(*this), end(*this), 0.0f);

        addBorder(1);
    }

    Matrix(Matrix&& m)
        : width { 0 }
        , height { 0 }
        , matrix { nullptr }
    {
        std::swap(*this, m);
    }

    reference operator()(size_t x, size_t y) {
#ifdef _DEBUG
        if (x >= width || y >= height) {
            throw std::out_of_range("Invalid Index");
        }
#endif

        return matrix[y * width + x];
    }

    value_type operator()(size_t x, size_t y) const {
#ifdef _DEBUG
        if (x >= width || y >= height) {
            throw std::out_of_range("Invalid Index");
        }
#endif

        return matrix[y * width + x];
    }

    size_t getWidth() const {
        return width;
    }

    size_t getHeight() const {
        return height;
    }

    pointer raw() {
        return matrix.get();
    }

    void addBorder(int w) {
        for (auto i = 0; i < w; ++i) {
            for (auto x = 0; x < width; ++x) {
                (*this)(x, i) = 1;
                (*this)(x, height - i - 1) = 1;
            }

            for (auto y = 0; y < height; ++y) {
                (*this)(i, y) = 1;
                (*this)(width - i - 1, y) = 1;
            }
        }
    }

private:
    size_t width, height;
    std::unique_ptr<value_type[]> matrix;

    friend pointer begin(this_type& m) {
        return m.matrix.get();
    }

    friend pointer end(this_type& m) {
        return &m.matrix[m.width * m.height];
    }

    friend std::ostream& operator<<(std::ostream& str, this_type const& m) {
        str << std::fixed << std::setprecision(2);

        for (auto y = 0; y < m.height; ++y) {
            for (auto x = 0; x < m.width; ++x) {
                str << m(x, y) << ' ';
            }

            str << '\n';
        }

        return str;
    }

    friend void swap(Matrix<T>& a, Matrix<T>& b) {
        std::swap(a.width, b.width);
        std::swap(a.height, b.height);
        std::swap(a.matrix, b.matrix);
    }
};
