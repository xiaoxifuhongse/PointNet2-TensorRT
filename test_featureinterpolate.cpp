#include <iostream>
#include<fstream>
#include "cuda_utils.h"
#include "featureinterpolatedlayer.h"
#include "logging.h"
using namespace  nvinfer1;
static Logger  gLogger;

const char* INPUT_BLOB_NAME1 = "xyz1";
const char* INPUT_BLOB_NAME2 = "xyz2";
const char* INPUT_BLOB_NAME3 = "points2";
const char* OUTPUT_BLOB_NAME = "output";

#define NUM (128)

IPluginV2Layer* featureInterpolate(INetworkDefinition* network, ITensor** inputs, int N, int M, int C)
{
    auto creator = getPluginRegistry()->getPluginCreator("FeatureInterpolatedPlugin_TRT", "1");
    std::vector<nvinfer1::PluginField>f;
    nvinfer1::PluginFieldCollection fc;
    f.emplace_back(PluginField("N",  &N,     PluginFieldType::kINT8, 1));

    f.emplace_back(PluginField("M",  &M,     PluginFieldType::kINT8, 1));

     f.emplace_back(PluginField("C",  &C,     PluginFieldType::kINT8, 1));


       fc.nbFields = f.size();
       fc.fields   = f.data();


    IPluginV2* pluginObj = creator->createPlugin("featureInterpolate", &fc);
    auto feature_interpolate_layer =network->addPluginV2(inputs, 3 ,*pluginObj);
    return feature_interpolate_layer;
}

void LogTensor(ITensor* layer,std::string msg=" ")
{
    std::cout<<msg<<":   ";
    for(int i=0; i<layer->getDimensions().nbDims; i++)
    {
        std::cout<<layer->getDimensions().d[i]<<" ";
    }
    std::cout<<std::endl;
}

ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt)
{
    INetworkDefinition* network = builder->createNetworkV2(0U);
    ITensor* xyz1    = network->addInput(INPUT_BLOB_NAME1, dt, Dims3{1,8000,3});//N
    ITensor* xyz2    = network->addInput(INPUT_BLOB_NAME2, dt, Dims3{1,512,3});//M
    ITensor* points2 = network->addInput(INPUT_BLOB_NAME3, dt, Dims3{1,512,6}); // C
    auto new_points = network->addShuffle(*points2);
    new_points->setFirstTranspose(Permutation{0,2,1});
    std::cout<<new_points->getOutput(0)->getDimensions().d[0]<<" "<<new_points->getOutput(0)->getDimensions().d[1]<<" "<<
                                                            new_points->getOutput(0)->getDimensions().d[2]<<std::endl;
    ITensor* input[] = {xyz1, xyz2,new_points->getOutput(0)};

    auto feature = featureInterpolate(network, input, 8000,512, 6);
    feature->getOutput(0)->setName(OUTPUT_BLOB_NAME);

    LogTensor(feature->getOutput(0),"feature_point");
    network->markOutput(*feature->getOutput(0));
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB

     std::cout << "Building engine, please wait for a while..." << std::endl;
     ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
     if(engine!=nullptr)
            std::cout << "Build engine successfully!" << std::endl;
     // Don't need the network any more
     network->destroy();
     return engine;
}


void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream)
{
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
}




#include <QFile>
#include <QTextStream>
int main()
{

    QFile  file_pt("xyz.csv");
     float xyz1  [1*8000*3];
     float xyz2  [1*512*3];
     float points[1*512*6];


     if(file_pt.open(QIODevice::ReadOnly))
     {
         QTextStream in_pt(&file_pt);
         int i=0;
         while(!in_pt.atEnd())
         {
               QString line = in_pt.readLine();
               QStringList strlist = line.split(",");
               inno_point pt;
              xyz1[i*3+0]  = strlist[0].toFloat();
              xyz1[i*3+1]  = strlist[1].toFloat();
              xyz1[i*3+2]  = strlist[2].toFloat();
              //std::cout<<i<<": "<<xyz1[i*3+0]<<" "<<xyz1[i*3+1]<<" "<<xyz1[i*3+1]<<std::endl;

              i++;
         }
     }


     QFile  file_pt_new("new_xyz.csv");

     if(file_pt_new.open(QIODevice::ReadOnly))
     {
         QTextStream in_pt(&file_pt_new);
         int i=0;
         while(!in_pt.atEnd())
         {
              QString line = in_pt.readLine();
              QStringList strlist = line.split(",");
              xyz2[i*3+0]  = strlist[0].toFloat();
              xyz2[i*3+1] = strlist[1].toFloat();
              xyz2[i*3+2] = strlist[2].toFloat();


              i++;

         }
     }

     QFile  file_pt_points("grouped_points_cat1.csv");

     if(file_pt_points.open(QIODevice::ReadOnly))
     {
         QTextStream in_pt(&file_pt_points);
         int i=0;
         while(!in_pt.atEnd())
         {
              QString line = in_pt.readLine();
              QStringList strlist = line.split(",");

              points[i*6+0]  = strlist[0].toFloat();
              points[i*6+1] = strlist[1].toFloat();
              points[i*6+2] = strlist[2].toFloat();
              points[i*6+3]  = strlist[3].toFloat();
              points[i*6+4] = strlist[4].toFloat();
              points[i*6+5] = strlist[5].toFloat();
//              std::cout<<i<<": "<<points[i*3+0]<<" "<<points[i*3+1]<<" "<<points[i*3+2]<<" "<<
//                         points[i*3+3]<<" "<<points[i*3+4]<<" "<<points[i*3+5]
//                      <<std::endl;
                if(i>510) break;
              i++;

         }
     }

    char *trtModelStream{nullptr};
    size_t size{0};
    IHostMemory* modelStream{nullptr};
   #if 0
       APIToModel(1, &modelStream);
       assert(modelStream != nullptr);
       std::ofstream p("feature.engine", std::ios::binary);
        if (!p) {
                 std::cerr << "could not open plan output file" << std::endl;
                 return -1;
             }
         p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
         modelStream->destroy();
   #else

     std::ifstream file("feature.engine", std::ios::binary);
       if (file.good())
       {
                 file.seekg(0, file.end);
                 size = file.tellg();
                 file.seekg(0, file.beg);
                 trtModelStream = new char[size];
                 assert(trtModelStream);
                 file.read(trtModelStream, size);
                 file.close();
        }
        IRuntime* runtime = createInferRuntime(gLogger);
        assert(runtime != nullptr);
        ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
        assert(engine != nullptr);
        IExecutionContext* context = engine->createExecutionContext();
        assert(context != nullptr);
        delete[] trtModelStream;
        void* buffers[4];
        const int inputIndex1 = engine->getBindingIndex(INPUT_BLOB_NAME1);
        const int inputIndex2 = engine->getBindingIndex(INPUT_BLOB_NAME2);
        const int inputIndex3 = engine->getBindingIndex(INPUT_BLOB_NAME3);
        const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);

        cudaMalloc(&buffers[inputIndex1],  1*3*8000*sizeof(float));
        cudaMalloc(&buffers[inputIndex2],  1*3*512*sizeof(float)) ;
        cudaMalloc(&buffers[inputIndex3],  1*6*512*sizeof(float)) ;
        cudaMalloc(&buffers[outputIndex],  1*8000*6*sizeof(float));
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        cudaMemcpyAsync(buffers[inputIndex1], xyz1,   1*3*8000*sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(buffers[inputIndex2], xyz2,   1*3*512*sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(buffers[inputIndex3], points, 1*6*512*sizeof(float), cudaMemcpyHostToDevice, stream);
        context->enqueue(1, buffers, stream, nullptr);
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
        cudaFree(buffers[inputIndex1]);
        cudaFree(buffers[inputIndex2]);
        cudaFree(buffers[inputIndex3]);
        cudaFree(buffers[outputIndex]);





   #endif


    return 1;
}
