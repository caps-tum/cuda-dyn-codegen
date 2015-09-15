#pragma once

#include <chrono>
#include <string>

class CpuTimer {
public:
    using clock = std::chrono::high_resolution_clock;
    using time_point = clock::time_point;
    using duration = std::chrono::microseconds;

    void start() {
        a = clock::now();
    }

    void stop() {
        b = clock::now();
    }

    duration getDuration() {
        return std::chrono::duration_cast<duration>(b - a);
    }

private:
    time_point a, b;
};

std::string to_string(std::chrono::microseconds us) {
    return std::to_string(us.count()) + " us";
}
