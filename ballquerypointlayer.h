#ifndef BALLQUERYPOINTLAYER_H
#define BALLQUERYPOINTLAYER_H

#include <iostream>
#include <vector>
#include "NvInfer.h"

namespace nvinfer1
{
    class BallQueryPointPlugin: public IPluginV2IOExt
    {
        public:

           explicit BallQueryPointPlugin();
           BallQueryPointPlugin(float radius_, int nSamples_, int K_, int N);                                 //2
           BallQueryPointPlugin(const void* data, size_t length);

           ~BallQueryPointPlugin();

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
               // return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kINT32;
            }
            void attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
            {

            };                    //19

            void configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput)
            {

            }           //20

            void detachFromContext()
            {

            };                                                                                          //21

        private:
            void forwardGPU( const float *const * inputs,int * output, cudaStream_t stream,int batchSize = 1);                            //22

            float radius;
            int   nSamples;
            int K;
            int N;
            const char* mPluginNamespace;
    };



    class BallQueryPointPluginCreator : public IPluginCreator
        {
            public:
                BallQueryPointPluginCreator();


                ~BallQueryPointPluginCreator() override = default;

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


                static PluginFieldCollection mFC;
                static std::vector<PluginField> mPluginAttributes;
        };
        REGISTER_TENSORRT_PLUGIN(BallQueryPointPluginCreator);

};

#endif // BALLQUERYPOINTLAYER_H
