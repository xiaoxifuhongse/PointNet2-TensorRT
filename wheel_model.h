#ifndef WHEEL_MODEL_H
#define WHEEL_MODEL_H
#include "cuda_utils.h"
#include "NvInfer.h"
#include "furthestpointsamplelayer.h"
#include "ballquerypointlayer.h"
#include "logging.h"
class WheelDetect
{

    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;
    cudaStream_t stream;
    void *buffers[2];

};

#endif // WHEEL_MODEL_H
