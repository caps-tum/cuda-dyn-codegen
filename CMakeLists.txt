cmake_minimum_required(VERSION 2.8.5)

set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")

project(cuda-dyn-codegen)

set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_DEBUG "-std=c++11 -g -D_DEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-std=c++11 -O2")

INCLUDE(FindCUDA)

set(CUDA_HOST_COMPILER "g++")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")

add_subdirectory(src)

