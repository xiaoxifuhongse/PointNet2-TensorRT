#ifndef FURTHESTPOINTSAMPLELAYER_H
#define FURTHESTPOINTSAMPLELAYER_H
#include <iostream>
#include <vector>
#include "NvInfer.h"

namespace nvinfer1
{
    class FurthestPointSamplePlugin: public IPluginV2IOExt
    {
        public:

           explicit FurthestPointSamplePlugin();                                     //1
           FurthestPointSamplePlugin(int num_, int nSamples_);                                 //2
           FurthestPointSamplePlugin(const void* data, size_t length);               //3

           ~FurthestPointSamplePlugin();                                             //4

            int getNbOutputs() const override
            {
                return 1;
            }

            Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;  //5

            int initialize() override;                                                           //6

            virtual void terminate() override {};

            virtual size_t getWorkspaceSize(int maxBatchSize) const override { return 0;}


            virtual size_t getSerializationSize() const override;                                                                           //8


            virtual void serialize(void* buffer) const override;                                                                            //9

            virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override;


            const char* getPluginType() const override;                                                                                 //10

            const char* getPluginVersion() const override;                                                                               //11

            void destroy() override;                                                                                                    //12

            IPluginV2IOExt* clone() const override;                                                                                     //13

            void setPluginNamespace(const char* pluginNamespace) override;                                                              //14

            const char* getPluginNamespace() const override;                                                                            //15

            DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;                   //16

            bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override;            //17

            bool canBroadcastInputAcrossBatch(int inputIndex) const override;
            //18
            bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const override
            {
                return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
            }
            void attachToContext(
                    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
            {

            };                    //19

            void configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput)
            {

            }           //20

            void detachFromContext()
            {

            };                                                                                          //21

        private:
            void forwardGpu( float  * inputs,float * output, cudaStream_t stream,int batchSize = 1);                            //22

            int nSamples;
          int NUM;

            const char* mPluginNamespace;
    };



class FurthestSamplePluginCreator : public IPluginCreator
{
public:
       FurthestSamplePluginCreator();


       ~FurthestSamplePluginCreator() override = default;

       const char* getPluginName() const override;

       const char* getPluginVersion() const override;

       const PluginFieldCollection* getFieldNames() override;

       IPluginV2IOExt* createPlugin(const char* name, const PluginFieldCollection* fc) override;

       IPluginV2IOExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

       void setPluginNamespace(const char* libNamespace) override
       {
           mNamespace = libNamespace;
       }

       const char* getPluginNamespace() const override
       {
           return mNamespace.c_str();
       }

private:
       std::string mNamespace;
       //int nSamples;
       static PluginFieldCollection mFC;
       static std::vector<PluginField> mPluginAttributes;
};
REGISTER_TENSORRT_PLUGIN(FurthestSamplePluginCreator);

};


#endif // FURTHESTPOINTSAMPLELAYER_H
