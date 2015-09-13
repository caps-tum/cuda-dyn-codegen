add_subdirectory(program_options)

set(gpu-impl_SRC
    CpuImpl.cpp
    CpuTimer.h
    Cuda.h
    CudaCall.h
    DynCodegen.cu
    GpuImpl.cu
    GpuTimer.h
    Logger.h
    main.cpp
    Matrix.h
    Stencil.h
    kernel_5p_v1.cu
    kernel_5p_v2.cu
    kernel_5p_v3.cu
    kernel_5p_v4.cu
    kernel_9p_v1.cu
    kernel_9p_v1_rt.h
)

include_directories(program_options)

cuda_add_executable(gpu-impl ${gpu-impl_SRC})

#target_link_libraries(gpu-impl cuda /home/hpc/pr87du/di73kos/cuda-7.0/lib64/libnvrtc.so)
#target_link_libraries(gpu-impl /usr/local/cuda-7.0/lib64/libnvrtc.so)

target_link_libraries(gpu-impl "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v7.0/lib/x64/nvrtc.lib")
target_link_libraries(gpu-impl "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v7.0/lib/x64/cuda.lib")

target_link_libraries(gpu-impl program_options)
