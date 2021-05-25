#include "cuda_utils.h"
#include <stdio.h>
#include <cassert>
#include <cstring>
#include "furthestpointsamplelayer.h"
using namespace nvinfer1;

__device__ void __update(float *__restrict__ dists, int *__restrict__ dists_i,
                         int idx1, int idx2) {
  const float v1 = dists[idx1], v2 = dists[idx2];
  const int i1 = dists_i[idx1], i2 = dists_i[idx2];
  dists[idx1] = max(v1, v2);
  dists_i[idx1] = v2 > v1 ? i2 : i1;
}


// Input dataset: (b, n, 3), tmp: (b, n)
// Ouput idxs (b, m)
template <unsigned int block_size>
__global__ void furthest_point_sampling_kernel(int b, int n, int m, const float *__restrict__ dataset, float *__restrict__ temp, int *__restrict__ idxs)
{

      if (m <= 0) return;
      __shared__ float dists[block_size];
      __shared__ int dists_i[block_size];

      int batch_index = blockIdx.x;
      dataset += batch_index * n * 3;
      temp += batch_index * n;
      idxs += batch_index * m;

      int tid = threadIdx.x;
      const int stride = block_size;

      int old = 0;
      if (threadIdx.x == 0) idxs[0] = old;

      __syncthreads();
      for (int j = 1; j < m; j++)
      {
        int besti = 0;
        float best = -1;
        float x1 = dataset[old * 3 + 0];
        float y1 = dataset[old * 3 + 1];
        float z1 = dataset[old * 3 + 2];
        for (int k = tid; k < n; k += stride)
        {
          float x2, y2, z2;
          x2 = dataset[k * 3 + 0];
          y2 = dataset[k * 3 + 1];
          z2 = dataset[k * 3 + 2];

          float mag = (x2 * x2) + (y2 * y2) + (z2 * z2);
          if (mag <= 1e-3) continue;

          float d =(x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);

          float d2 = min(d, temp[k]);
          temp[k] = d2;
          besti = d2 > best ? k : besti;
          best = d2 > best ? d2 : best;
        }
        dists[tid] = best;
        dists_i[tid] = besti;
        __syncthreads();

        if (block_size >= 512) {
          if (tid < 256) {
            __update(dists, dists_i, tid, tid + 256);
          }
          __syncthreads();
        }
        if (block_size >= 256) {
          if (tid < 128) {
            __update(dists, dists_i, tid, tid + 128);
          }
          __syncthreads();
        }
        if (block_size >= 128) {
          if (tid < 64) {
            __update(dists, dists_i, tid, tid + 64);
          }
          __syncthreads();
        }
        if (block_size >= 64) {
          if (tid < 32) {
            __update(dists, dists_i, tid, tid + 32);
          }
          __syncthreads();
        }
        if (block_size >= 32) {
          if (tid < 16) {
            __update(dists, dists_i, tid, tid + 16);
          }
          __syncthreads();
        }
        if (block_size >= 16) {
          if (tid < 8) {
            __update(dists, dists_i, tid, tid + 8);
          }
          __syncthreads();
        }
        if (block_size >= 8) {
          if (tid < 4) {
            __update(dists, dists_i, tid, tid + 4);
          }
          __syncthreads();
        }
        if (block_size >= 4) {
          if (tid < 2) {
            __update(dists, dists_i, tid, tid + 2);
          }
          __syncthreads();
        }
        if (block_size >= 2) {
          if (tid < 1) {
            __update(dists, dists_i, tid, tid + 1);
          }
          __syncthreads();
        }

        old = dists_i[0];
        if (tid == 0) idxs[j] = old;
      }
}

void furthest_point_sampling_kernel_wrapper(int b, int n, int m,
                                              float *dataset, float *temp,
                                            int *idxs) {
  unsigned int n_threads = opt_n_threads(n);

 // cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  switch (n_threads) {
    case 512:
      furthest_point_sampling_kernel<512>
          <<<b, n_threads>>>(b, n, m, dataset, temp, idxs);
      break;
    case 256:
      furthest_point_sampling_kernel<256>
          <<<b, n_threads>>>(b, n, m, dataset, temp, idxs);
      break;
    case 128:
      furthest_point_sampling_kernel<128>
          <<<b, n_threads>>>(b, n, m, dataset, temp, idxs);
      break;
    case 64:
      furthest_point_sampling_kernel<64>
          <<<b, n_threads>>>(b, n, m, dataset, temp, idxs);
      break;
    case 32:
      furthest_point_sampling_kernel<32>
          <<<b, n_threads>>>(b, n, m, dataset, temp, idxs);
      break;
    case 16:
      furthest_point_sampling_kernel<16>
          <<<b, n_threads>>>(b, n, m, dataset, temp, idxs);
      break;
    case 8:
      furthest_point_sampling_kernel<8>
          <<<b, n_threads>>>(b, n, m, dataset, temp, idxs);
      break;
    case 4:
      furthest_point_sampling_kernel<4>
          <<<b, n_threads>>>(b, n, m, dataset, temp, idxs);
      break;
    case 2:
      furthest_point_sampling_kernel<2>
          <<<b, n_threads>>>(b, n, m, dataset, temp, idxs);
      break;
    case 1:
      furthest_point_sampling_kernel<1>
          <<<b, n_threads>>>(b, n, m, dataset, temp, idxs);
      break;
    default:
      furthest_point_sampling_kernel<512>
          <<<b, n_threads >>>(b, n, m, dataset, temp, idxs);
  }

  CUDA_CHECK_ERRORS();
}


// input: points(b, c, n) idx(b, m)
// output: out(b, c, m)
__global__ void gather_points_kernel(int b, int c, int n, int m,
                                     const float *__restrict__ points,
                                     const int *__restrict__ idx,
                                     float *__restrict__ out) {

      for (int i = blockIdx.x; i < b; i += gridDim.x) {
        for (int l = blockIdx.y; l < c; l += gridDim.y) {
          for (int j = threadIdx.x; j < m; j += blockDim.x) {

            int a = idx[i * m + j];

            out[3*j+l] = points[3*a + l];
          }
        }
      }


}

void gather_points_kernel_wrapper(int b, int c, int n, int npoints,
                                  const float *points, const int *idx,
                                  float *out) {
   // printf("%d,%d,%d,%d,%d\n",b,c,n,npoints,opt_n_threads(npoints));
  gather_points_kernel<<<dim3(b, c, 1), opt_n_threads(npoints)>>>(b, c, n, npoints,points, idx, out);

  CUDA_CHECK_ERRORS();
}



namespace {
    static const char* FURHTEST_SAMPLE_PLUGIN_VERSION{"1"};
    static const char* FURHTEST_SAMPLE_PLUGIN_NAME{"FurthestPointSamplePlugin"};
}



FurthestPointSamplePlugin::FurthestPointSamplePlugin()
{

}

FurthestPointSamplePlugin::FurthestPointSamplePlugin(int num_, int nSamples_)
{
    NUM = num_;
    nSamples = nSamples_;

}

FurthestPointSamplePlugin::~FurthestPointSamplePlugin()
{


}

FurthestPointSamplePlugin::FurthestPointSamplePlugin(const void *data, size_t length)
{
    const char*d = reinterpret_cast<const char*>(data), *a = d;
    NUM = readFromBuffer<int>(d);
    nSamples = readFromBuffer<int>(d);

    assert(d==a+length);
}




Dims FurthestPointSamplePlugin::getOutputDimensions(int index, const Dims *inputs, int nbInputDims)
{
    //printf("====%d,%d,%d\n",inputs[0].d[0],inputs[0].d[1],inputs[0].d[2]);
    return Dims3{inputs[0].d[0],nSamples,inputs[0].d[2] };
}


int FurthestPointSamplePlugin::initialize()
{
    return 0;
}


void FurthestPointSamplePlugin::serialize(void *buffer) const
{
    char *d = static_cast<char*>(buffer);
    const char *a = d;
     writeToBuffer(d, NUM);
     writeToBuffer(d, nSamples);
    assert(d==a+getSerializationSize());
}

size_t FurthestPointSamplePlugin::getSerializationSize() const
{
    return sizeof(nSamples)+sizeof(NUM);
}



void  FurthestPointSamplePlugin::setPluginNamespace(const char* pluginNamespace)
 {
     mPluginNamespace =  pluginNamespace;
 }

const char* FurthestPointSamplePlugin::getPluginNamespace() const
{
    return mPluginNamespace;
}

DataType FurthestPointSamplePlugin::getOutputDataType(int index, const DataType *inputTypes, int nbInputs) const
{
    return DataType::kFLOAT;
}


bool FurthestPointSamplePlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
    return false;
}

bool FurthestPointSamplePlugin::canBroadcastInputAcrossBatch(int inputIndex) const
{ //printf("====%d,%d,%d\n",inputs[0].d[0],inputs[0].d[1],inputs[0].d[2]);
    return false;
}



const char* FurthestPointSamplePlugin::getPluginType() const
{
    return "FursthestPointSample_TRT";
}


const char* FurthestPointSamplePlugin::getPluginVersion() const
{
    return "1";
}

void FurthestPointSamplePlugin::destroy()
{
    delete this;
}

// Clone the plugin
IPluginV2IOExt* FurthestPointSamplePlugin::clone() const
{
       FurthestPointSamplePlugin *p = new FurthestPointSamplePlugin(nSamples,nSamples);
       p->setPluginNamespace(mPluginNamespace);
       return p;
}


int FurthestPointSamplePlugin::enqueue(int batchSize, const void * const *inputs, void **outputs, void *workspace, cudaStream_t stream)
{
      forwardGpu(( float *)inputs[0], (float*)outputs[0], stream, batchSize);
}


void FurthestPointSamplePlugin::forwardGpu( float  *inputs, float *output, cudaStream_t stream, int batchSize)
{

    float *d_temp;
    cudaMalloc((void**)&d_temp,NUM*sizeof(float));
    cudaMemset(d_temp, 1e10, NUM*sizeof(float));

    int *d_output;
    cudaMalloc((void**)&d_output, nSamples*sizeof(int)) ;
    cudaMemset(d_output,0,nSamples*sizeof(int));

    furthest_point_sampling_kernel_wrapper(batchSize,NUM, nSamples, inputs, d_temp,d_output);

//    int *out = new int[nSamples];
//    cudaMemcpyAsync(out, d_output, 1 * nSamples * sizeof(int), cudaMemcpyDeviceToHost);
//   for(int i=0; i<nSamples; i++)
//    printf("%d\n",out[i]);

    gather_points_kernel_wrapper(batchSize, 3, NUM, nSamples, inputs, d_output,output);
    
    cudaFree(d_temp);
    cudaFree(d_output);
}




PluginFieldCollection FurthestSamplePluginCreator::mFC{};
std::vector<PluginField> FurthestSamplePluginCreator::mPluginAttributes;

FurthestSamplePluginCreator::FurthestSamplePluginCreator()
{
    mPluginAttributes.clear();
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* FurthestSamplePluginCreator::getPluginName() const
{
     return "FursthestPointSample_TRT";
}

const char* FurthestSamplePluginCreator::getPluginVersion() const
{
     return "1";
}

const PluginFieldCollection* FurthestSamplePluginCreator::getFieldNames()
{

      return &mFC;
}

IPluginV2IOExt* FurthestSamplePluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
     int num_    =  *(int*)(fc->fields[0].data);
     int samples_ = *(int*)(fc->fields[1].data);
     FurthestPointSamplePlugin* obj = new FurthestPointSamplePlugin(num_,samples_);
     obj->setPluginNamespace(mNamespace.c_str());
     return obj;
}

IPluginV2IOExt* FurthestSamplePluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
        // This object will be deleted when the network is destroyed, which will
        // call MishPlugin::destroy()
        FurthestPointSamplePlugin* obj = new FurthestPointSamplePlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
}



