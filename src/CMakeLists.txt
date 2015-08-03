set(cpu-impl_SRC
    CpuImpl.cpp
)

set(gpu-impl_SRC
    GpuImpl.cu
    #matrixtest.cu
)

set(dyn-codegen_SRC
    DynCodegen.cu
)

#cuda_add_executable(cpu-impl ${cpu-impl_SRC})
cuda_add_executable(gpu-impl ${gpu-impl_SRC})

#cuda_add_executable(dyn-codegen ${dyn-codegen_SRC})
#target_link_libraries(dyn-codegen cuda /home/hpc/pr87du/di73kos/cuda-7.0/lib64/libnvrtc.so)
#target_link_libraries(dyn-codegen /usr/local/cuda-7.0/lib64/libnvrtc.so)

