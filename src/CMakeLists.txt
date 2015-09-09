add_subdirectory(program_options)

set(cpu-impl_SRC
    CpuImpl.cpp
)

set(gpu-impl_SRC
    Cuda.h
    CudaCall.h
    GpuImpl.cu
    GpuTimer.h
    Logger.h
    Matrix.h
    Stencil.h
    Test.h
)

set(dyn-codegen_SRC
    DynCodegen.cu
)

include_directories(program_options)

#cuda_add_executable(cpu-impl ${cpu-impl_SRC})
#cuda_add_executable(gpu-impl ${gpu-impl_SRC})

cuda_add_executable(dyn-codegen ${dyn-codegen_SRC})
#target_link_libraries(dyn-codegen cuda /home/hpc/pr87du/di73kos/cuda-7.0/lib64/libnvrtc.so)
#target_link_libraries(dyn-codegen /usr/local/cuda-7.0/lib64/libnvrtc.so)

target_link_libraries(dyn-codegen "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v7.0/lib/x64/nvrtc.lib")
target_link_libraries(dyn-codegen "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v7.0/lib/x64/cuda.lib")

#target_link_libraries(gpu-impl program_options)

