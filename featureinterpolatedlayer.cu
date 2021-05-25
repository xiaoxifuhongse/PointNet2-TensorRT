#include <stdio.h>
#include <cassert>
#include <math.h>
#include <stdlib.h>
#include <cstring>
#include "featureinterpolatedlayer.h"
#include "cuda_utils.h"
using namespace nvinfer1;

// input: unknown(b, n, 3) known(b, m, 3)
// output: dist2(b, n, 3), idx(b, n, 3)
__global__ void three_nn_kernel(int b, int n, int m,
                                const float *__restrict__ xyz1,  // unknow
                                const float *__restrict__ xyz2,    //known
                                float *__restrict__ dist2,
                                int *__restrict__ idx) 
{
    int batch_index = blockIdx.x;
    xyz1  += batch_index * n * 3;// n= 512
    xyz2  += batch_index * m * 3; // m 128
 	dist2 += batch_index * n * 3;
 	idx   += batch_index * n * 3;

    int index  = threadIdx.x;
    int stride = blockDim.x;
    for (int j = index; j < n; j += stride) 
    {
	    float ux = xyz1[j * 3 + 0];
	    float uy = xyz1[j * 3 + 1];
	    float uz = xyz1[j * 3 + 2];

	    double best1 = 1e40, best2 = 1e40, best3 = 1e40;
	    int besti1 = 0, besti2 = 0, besti3 = 0;
	    for(int k = 0; k < m; ++k) 
	    {
		    float x = xyz2[k * 3 + 0];
		    float y = xyz2[k * 3 + 1];
		    float z = xyz2[k * 3 + 2];

		    float d = (ux - x) * (ux - x) + (uy - y) * (uy - y) + (uz - z) * (uz - z);
           if (d < best1) 
	       {
		        best3 = best2;
		        besti3 = besti2;
		        best2 = best1;
		        besti2 = besti1;
		        best1 = d;
		        besti1 = k;
      		}
      		else if (d < best2)
      		{
		        best3 = best2;
		        besti3 = besti2;
		        best2 = d;
		        besti2 = k;
      		}
	       else if (d < best3) 
	       {
		        best3 = d;
		        besti3 = k;
	       }
   		}

        float norm = 1/(best1+1e-8)+ 1/(best2+1e-8)+1/(best3+1e-8);
        dist2[j * 3 + 0] = (1/(best1+1e-8))/norm;
        dist2[j * 3 + 1] = (1/(best2+1e-8))/norm;
        dist2[j * 3 + 2] = (1/(best3+1e-8))/norm;

	    idx[j * 3 + 0] = besti1;
	    idx[j * 3 + 1] = besti2;
	    idx[j * 3 + 2] = besti3;
  	}
}

void three_nn_kernel_wrapper(int b, int n, int m, const float *xyz1,
                             const float *xyz2, float *weight, int *idx)
{

  three_nn_kernel<<<b, opt_n_threads(n), 0>>>(b, n, m, xyz1, xyz2, weight, idx);
  CUDA_CHECK_ERRORS();
}

// input: points(b, c, m), idx(b, n, 3), weight(b, n, 3)
// output: out(b, c, n)
__global__ void three_interpolate_kernel(int b, int c, int m, int n,
                                         const float *__restrict__ points,
                                         const int *__restrict__ idx,
                                         const float *__restrict__ weight,
                                         float *__restrict__ out)
{

    int batch_index = blockIdx.x;

    points += batch_index * m * c;

    idx += batch_index * n * 3;
    weight += batch_index * n * 3;

    out += batch_index * n * c;

    const int index = threadIdx.y * blockDim.x + threadIdx.x;
    const int stride = blockDim.y * blockDim.x;


    for (int i = index; i < c * n; i += stride)
    {
        const int l = i / n;
        const int j = i % n;

        float w1 = weight[j * 3 + 0];
        float w2 = weight[j * 3 + 1];
        float w3 = weight[j * 3 + 2];

        int i1 = idx[j * 3 + 0];
        int i2 = idx[j * 3 + 1];
        int i3 = idx[j * 3 + 2];
        //printf("[%d %f %f]",  j,points[l * m + i2],points[l * m + i3]);
        out[i] =

                 points[l * m + i1] * w1 + points[l * m + i2] * w2 +
                 points[l * m + i3] * w3;
  }
}

void three_interpolate_kernel_wrapper(int b, int c, int m, int n,
                                      const float *points, const int *idx,
                                      const float *weight, float *out) 
{
   three_interpolate_kernel<<<b, opt_block_config(n, c), 0>>>(b, c, m, n, points, idx, weight, out);

   CUDA_CHECK_ERRORS();
}


/********************************


********************************/

FeatureInterpolatedPlugin::FeatureInterpolatedPlugin()
{

}

FeatureInterpolatedPlugin::FeatureInterpolatedPlugin(int N_, int M_, int C_)
{
    N  = N_;

    M  = M_;

    C  = C_;


}


FeatureInterpolatedPlugin::FeatureInterpolatedPlugin(const void* data, size_t length)
{
    const char*d = reinterpret_cast<const char*>(data), *a = d;
    N   = readFromBuffer<int>(d);

    M   = readFromBuffer<int>(d);

    C   = readFromBuffer<int>(d);
  

    assert(d==a+length);

}


FeatureInterpolatedPlugin::~FeatureInterpolatedPlugin()
{

}

int FeatureInterpolatedPlugin::initialize()
{
    return 0;
}

Dims FeatureInterpolatedPlugin::getOutputDimensions(int index, const Dims* inputs, int getNbOutputs)
{
//    printf("N: %d\n", N);
//    printf("M: %d\n", M);
//    printf("C: %d\n", C);
    return Dims3(1, C, N );
}

void FeatureInterpolatedPlugin::serialize(void* buffer) const
{
    char *d = static_cast<char*>(buffer);
    const char *a = d;
    writeToBuffer(d, N);
    writeToBuffer(d, M);
    writeToBuffer(d, C);

    assert(d==a+getSerializationSize());
}

size_t FeatureInterpolatedPlugin::getSerializationSize() const
{
     return sizeof(N)+sizeof(M)+sizeof(C);
}


void FeatureInterpolatedPlugin::setPluginNamespace(const char* pluginNamespace)
{
	mPluginNamespace = pluginNamespace;
}


const char* FeatureInterpolatedPlugin::getPluginNamespace() const 
{
	return mPluginNamespace;
}

DataType FeatureInterpolatedPlugin::getOutputDataType(int index, const DataType*inputTypes, int nbInputs) const
{
	return DataType::kFLOAT;
}



bool FeatureInterpolatedPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
    return false;
}

bool FeatureInterpolatedPlugin::canBroadcastInputAcrossBatch(int inputIndex) const
{
    return false;
}

const char* FeatureInterpolatedPlugin::getPluginType() const
{
    return "FeatureInterpolatedPlugin_TRT";
}


const char* FeatureInterpolatedPlugin::getPluginVersion() const
{
    return "1";
}

void FeatureInterpolatedPlugin::destroy()
{
    delete this;
}


// Clone the plugin
IPluginV2IOExt* FeatureInterpolatedPlugin::clone() const
{
    FeatureInterpolatedPlugin *p = new FeatureInterpolatedPlugin(N,M,C);
    p->setPluginNamespace(mPluginNamespace);
    return p;
}



int FeatureInterpolatedPlugin::enqueue(int batchSize, const void*const * inputs, void **outputs, void *workspace, cudaStream_t stream)
{

    forwardGPU((const float *const *)inputs, (float*)outputs[0], stream, batchSize);

    return 0;

}


void FeatureInterpolatedPlugin::forwardGPU(const float* const* inputs, float *output, cudaStream_t stream, int batchSize)
{

    int* d_idx; // 1*N*3
    float* d_weight;// 1*N*3

    cudaMalloc((void**)&d_idx,   1*N*3*sizeof(int));
    cudaMalloc((void**)&d_weight,1*N*3*sizeof(float));

    three_nn_kernel_wrapper(1,N,M,inputs[0],inputs[1],d_weight, d_idx);
    three_interpolate_kernel_wrapper(1,C,M,N,inputs[2],d_idx, d_weight, output);



//     float *h_output = new float[1*C*N];

//     cudaMemcpy(h_output,output,1*C*N*sizeof(float),cudaMemcpyDeviceToHost);
//    for(int i=0; i<C; i++)
//    {  //printf("===========%d=================\n",i);
//        for(int j=0; j<1; j++)
//        {
//            printf("%f ", h_output[i*N+j]);
//        }
//        //printf("\n");

//    }

     cudaFree(d_weight);
     cudaFree(d_idx);
     CUDA_CHECK_ERRORS();
}



/********************************************************

********************************************************/

PluginFieldCollection FeatureInterpolatedPluginCreator::mFC{};
std::vector<PluginField> FeatureInterpolatedPluginCreator::mPluginAttributes;

FeatureInterpolatedPluginCreator::FeatureInterpolatedPluginCreator()
{
	mPluginAttributes.clear();
	mFC.nbFields = mPluginAttributes.size();
	mFC.fields   = mPluginAttributes.data();
}


const char* FeatureInterpolatedPluginCreator::getPluginName() const
{

	return "FeatureInterpolatedPlugin_TRT";
}


const char* FeatureInterpolatedPluginCreator::getPluginVersion() const
{
	return "1";
}

const PluginFieldCollection* FeatureInterpolatedPluginCreator::getFieldNames()
{
	return &mFC;
}


IPluginV2IOExt* FeatureInterpolatedPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{

    int N     = *(int*)(fc->fields[0].data);

    int M     = *(int*)(fc->fields[1].data);

    int C     = *(int*)(fc->fields[2].data);


    FeatureInterpolatedPlugin* obj = new FeatureInterpolatedPlugin(N,M,C);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}


IPluginV2IOExt* FeatureInterpolatedPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    FeatureInterpolatedPlugin* obj = new FeatureInterpolatedPlugin(serialData, serialLength);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}





