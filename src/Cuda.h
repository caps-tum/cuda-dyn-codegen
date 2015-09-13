#pragma once

#include <iostream>
#include <memory>

#include <cuda.h>
#include <cuda_runtime.h>

struct Cuda {
    CUdevice device;
    CUcontext context;

    Cuda() {
        cuInit(0);

        cuDeviceGet(&device, 0);
        cuCtxCreate(&context, 0, device);
    }

    virtual ~Cuda() {
        cuCtxDestroy(context);
    }

    CUdevice& getDevice() {
        return device;
    }

    CUcontext& getContext() {
        return context;
    }
};
