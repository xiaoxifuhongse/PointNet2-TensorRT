#include "cuda_utils.h"
#include <stdio.h>
#include <cassert>
#include <cstring>
#include "grouppointslayer.h"
#include <fcntl.h>
using namespace nvinfer1;
// input: points(b, c, n) idx(b, npoints, nsample)
// output: out(b, c, npoints, nsample)
__global__ void group_points_kernel(int b, int c, int n, int npoints,int nsample,
                                    const float *__restrict__ points,
                                    const int *__restrict__ idx,
                                    float*__restrict__ grouped_points)
{



    int batch_index  = blockIdx.x;

    points          += batch_index *n * c;
    idx             += batch_index * npoints * nsample; // npoints:128  nsample: 64
    grouped_points  += batch_index * npoints * nsample * c;

    const int index  = threadIdx.y * blockDim.x + threadIdx.x;
    const int stride = blockDim.y * blockDim.x;

    for (int i = index; i < c * npoints; i += stride)
    {
      const int l = i / npoints;
      const int j = i % npoints;

      for (int k = 0; k < nsample; ++k)
      {
        int ii = idx[j * nsample + k];
         //printf("%d  ",(ii*c+l)<n*c);

       grouped_points[(l * npoints + j) * nsample + k]   =points[ii * c + l];

      }
    }
}


void group_points_kernel_wrapper(int b, int c, int n, int npoints, int nsample,
                                const  float *points,  const int *idx,
                                 float *out)
{

//      printf("n:%d\n", n);
//      printf("npoints:%d\n", npoints);
//      printf("nsample:%d\n", nsample);

    group_points_kernel<<<b, opt_block_config(npoints, c), 0>>>(b, c, n, npoints, nsample, points, idx, out);
    CUDA_CHECK_ERRORS();
}


//int GroupPointPlugin::N = 0;
//int GroupPointPlugin::C = 0;
//int GroupPointPlugin::npoints = 0;
//int GroupPointPlugin::nsamples = 0;

GroupPointPlugin::GroupPointPlugin()
{

}

GroupPointPlugin::GroupPointPlugin(int n_, int c_, int npoints_, int nsamples_)
{
  N = n_;
  C = c_;
  npoints = npoints_;
  nsamples = nsamples_;
}

GroupPointPlugin::~GroupPointPlugin()
{

}

int GroupPointPlugin::initialize()
{
    return 0;
}


GroupPointPlugin::GroupPointPlugin(const void *data, size_t length)
{
    const char*d = reinterpret_cast<const char*>(data), *a = d;
    N        = readFromBuffer<int>(d);
    C        = readFromBuffer<int>(d);
    nsamples = readFromBuffer<int>(d);
    npoints  = readFromBuffer<int>(d);

    assert(d==a+length);
}


Dims GroupPointPlugin::getOutputDimensions(int index, const Dims*inputs, int getNbOutputs)
{

    return Dims4{1, C, npoints,nsamples }; //1*c*npoints*K 1*3*512*32
}


void GroupPointPlugin::serialize(void *buffer) const
{
    char *d = static_cast<char*>(buffer);
    const char *a = d;
    writeToBuffer(d, N);
    writeToBuffer(d,C);
    writeToBuffer(d,nsamples);
    writeToBuffer(d,npoints);

    assert(d==a+getSerializationSize());
}


size_t GroupPointPlugin::getSerializationSize() const
{
    return sizeof(N)+ sizeof(C)+ sizeof(nsamples)+sizeof(npoints);
}


void  GroupPointPlugin::setPluginNamespace(const char* pluginNamespace)
 {
     mPluginNamespace =  pluginNamespace;
 }


const char* GroupPointPlugin::getPluginNamespace() const
{
    return mPluginNamespace;
}

DataType GroupPointPlugin::getOutputDataType(int index, const DataType *inputTypes, int nbInputs) const
{
    return DataType::kFLOAT;
}

bool GroupPointPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
    return false;
}

bool GroupPointPlugin::canBroadcastInputAcrossBatch(int inputIndex) const
{
    return false;
}

const char* GroupPointPlugin::getPluginType() const
{
    return "GroupPointPlugin_TRT";
}



const char* GroupPointPlugin::getPluginVersion() const
{
    return "1";
}

void GroupPointPlugin::destroy()
{
    delete this;
}


// Clone the plugin
IPluginV2IOExt* GroupPointPlugin::clone() const
{
    GroupPointPlugin *p = new GroupPointPlugin(N,C,npoints, nsamples);
    p->setPluginNamespace(mPluginNamespace);
    return p;
}



int GroupPointPlugin::enqueue(int batchSize, const void*const * inputs, void **outputs, void *workspace, cudaStream_t stream)
{

    const float* points = static_cast<const float *>(inputs[0]);
    const int* index = static_cast<const int *>(inputs[1]);
      float* out = static_cast< float *>(outputs[0]);
     forwardGPU( points,index, out, stream, batchSize);
    return 0;

}


void GroupPointPlugin::forwardGPU( const float* points, const int* idx, float*output,  cudaStream_t stream, int batchSize)
{

//    float *tmp;
//    cudaMalloc(&tmp,sizeof(float)*C*N);
//    printf("%d,%d\n",C,N);
//    cudaMemcpyAsync(tmp, points,C*N*sizeof(float),cudaMemcpyDeviceToDevice,stream);
    group_points_kernel_wrapper(batchSize, C, N, npoints, nsamples, points, idx, output);




}





PluginFieldCollection GroupPointPluginCreator::mFC{};
std::vector<PluginField> GroupPointPluginCreator::mPluginAttributes;

GroupPointPluginCreator::GroupPointPluginCreator()
{
    mPluginAttributes.clear();
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}



const char* GroupPointPluginCreator::getPluginName() const
{
     return "GroupPointPlugin_TRT";
}

const char* GroupPointPluginCreator::getPluginVersion() const
{
     return "1";
}

const PluginFieldCollection* GroupPointPluginCreator::getFieldNames()
{

      return &mFC;
}

IPluginV2IOExt* GroupPointPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    int N  = *(int*)(fc->fields[0].data);
    int C         = *(int*)(fc->fields[1].data);
    int npoints         =  *(int*)(fc->fields[2].data);
    int nsamples     = *(int*)(fc->fields[3].data);

    GroupPointPlugin* obj = new GroupPointPlugin(N, C,npoints, nsamples);

    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}

IPluginV2IOExt* GroupPointPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    GroupPointPlugin* obj = new GroupPointPlugin(serialData, serialLength);

    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}

