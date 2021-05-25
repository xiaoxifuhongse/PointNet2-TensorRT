#ifndef FEATUREINTERPOLATEDLAYER_H
#define FEATUREINTERPOLATEDLAYER_H
#include <iostream>
#include <vector>
#include "NvInfer.h"

namespace nvinfer1
{

	class FeatureInterpolatedPlugin:public IPluginV2IOExt
	{
	public:
		explicit FeatureInterpolatedPlugin();
        FeatureInterpolatedPlugin(int N_,  int M_, int C_);
		FeatureInterpolatedPlugin(const void* data, size_t length);
    
		~FeatureInterpolatedPlugin();
		
		int getNbOutputs() const override
		{
			return 1;
		}

		Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

		int initialize() override;

		virtual void terminate() override { };

		virtual size_t getWorkspaceSize(int maxBatchSize) const override {return 0;}

		virtual size_t getSerializationSize() const override;

		virtual void   serialize(void* buffer) const override;

		virtual int enqueue(int batchSize, const void* const* input, void** outputs, void* workspace, cudaStream_t stream) override;

		const char* getPluginType() const override;

		const char* getPluginVersion() const override;

        void destroy() override;

		IPluginV2IOExt* clone() const override;

		void setPluginNamespace(const char* pluginNamespace) override;

		const char* getPluginNamespace() const override;

		DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;                   //16

        bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override;            //17

        bool canBroadcastInputAcrossBatch(int inputIndex) const override;
            
        bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const override
        {
            return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
        }

        void attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
        {

        };                    

        void configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput)
        {
//           std::cout<<in[0].dims.d[0]<<"  :  "<<in[0].dims.d[1]<<"  : "<<in[0].dims.d[2]<<std::endl;
//            std::cout<<in[1].dims.d[0]<<"  :  "<<in[1].dims.d[1]<<"  : "<<in[1].dims.d[2]<<std::endl;
//             std::cout<<in[2].dims.d[0]<<"  :  "<<in[2].dims.d[1]<<"  : "<<in[2].dims.d[2]<<std::endl;
//             std::cout<<"============"<<std::endl;
             N = in[0].dims.d[1];
             M= in[1].dims.d[1];
             C =in[2].dims.d[1];
        }           

        void detachFromContext()
        {

        };     


    private:

         void forwardGPU( const float *const * inputs,float * output, cudaStream_t stream,int batchSize = 1);

         const char* mPluginNamespace;
         int N;
         int M;
         int C;



	}; // class





	class FeatureInterpolatedPluginCreator: public IPluginCreator
	{
	public:
		FeatureInterpolatedPluginCreator();
		~FeatureInterpolatedPluginCreator() override = default;

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

	};// class

	REGISTER_TENSORRT_PLUGIN(FeatureInterpolatedPluginCreator);




};// namespace



#endif // FEATUREINTERPOLATEDLAYER_H
