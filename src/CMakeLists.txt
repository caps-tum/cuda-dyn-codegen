if(NOT MAC)
  add_subdirectory(program_options)
endif()

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
    kernel_5p_v5.cu
    kernel_9p_v1.cu
)

if(MAC)
  include_directories($ENV{BOOST_INCDIR})
else()
  include_directories(program_options)
endif()

cuda_add_executable(gpu-impl ${gpu-impl_SRC})

if(IVY)
  target_link_libraries(gpu-impl /usr/local/cuda-7.0/lib64/libnvrtc.so)
  target_link_libraries(gpu-impl program_options)
elseif(MAC)
  target_link_libraries(gpu-impl /home/hpc/pr87du/di73kos/extr/NVIDIA-Linux-x86_64-346.46/libcuda.so.346.46 /home/hpc/pr87du/di73kos/cuda-7.0/lib64/libnvrtc.so)
  target_link_libraries(gpu-impl $ENV{BOOST_LIBDIR}/libboost_program_options.a)
else()
  target_link_libraries(gpu-impl "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v7.0/lib/x64/nvrtc.lib")
  target_link_libraries(gpu-impl "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v7.0/lib/x64/cuda.lib")
  target_link_libraries(gpu-impl program_options)
endif()

