cmake_minimum_required(VERSION 2.8.5)

if(DEFINED IVY)
  set(CMAKE_C_COMPILER "gcc")
  set(CMAKE_CXX_COMPILER "g++")

elseif(DEFINED MAC)
else()
endif()

project(cuda-dyn-codegen)

if(DEFINED IVY)
  set(CMAKE_CXX_FLAGS "-std=c++11")
  set(CMAKE_CXX_FLAGS_DEBUG "-std=c++11 -g -D_DEBUG")
  set(CMAKE_CXX_FLAGS_RELEASE "-std=c++11 -O2")

  set(CUDA_HOST_COMPILER "g++")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")

elseif(DEFINED MAC)
  set(CMAKE_CXX_FLAGS "-std=c++11")
  set(CMAKE_CXX_FLAGS_DEBUG "-std=c++11 -g -D_DEBUG")
  set(CMAKE_CXX_FLAGS_RELEASE "-std=c++11 -O2")

else()

endif()

find_package(CUDA REQUIRED)

set(CMAKE_INCLUDE_DIRECTORIES_BEFORE ON)

add_subdirectory(src)

