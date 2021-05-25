#include <iostream>
#include <QFile>
#include <QTextStream>
#include "cuda_utils.h"
#include "NvInfer.h"
#include "furthestpointsamplelayer.h"
#include "ballquerypointlayer.h"
#include "logging.h"
#include <fcntl.h>
#include<fstream>
using namespace nvinfer1;
static Logger gLogger;
#define NUM  8000
#define C    3
const char* INPUT_BLOB_NAME = "xyz";
const char* OUTPUT_BLOB_NAME = "seg";

void LogTensor(ITensor* layer,std::string msg=" ")
{
    std::cout<<msg<<":   ";
    for(int i=0; i<layer->getDimensions().nbDims; i++)
    {
        std::cout<<layer->getDimensions().d[i]<<" ";
    }
    std::cout<<std::endl;
}


ILayer* query_ball_point(INetworkDefinition *network, ITensor** inputs, int N, int nPoint, int nSample,float radius)
{
    auto creator                          = getPluginRegistry()->getPluginCreator("BallQueryPointPlugin_TRT", "1");
    std::vector<nvinfer1::PluginField>f;
    nvinfer1::PluginFieldCollection fc;
    f.emplace_back(PluginField("nSamples",  &nPoint,   PluginFieldType::kINT8, 1));
    f.emplace_back(PluginField("K",         &nSample,  PluginFieldType::kINT8, 1));
    f.emplace_back(PluginField("radius",    &radius,   PluginFieldType::kFLOAT16, 1));
    f.emplace_back(PluginField("N",         &N,    PluginFieldType::kINT8, 1));

    fc.nbFields = f.size();
    fc.fields   = f.data();
    IPluginV2 *pluginObj = creator->createPlugin("ballquerylayer", &fc);
    auto queryballpointlayer = network->addPluginV2(inputs, 2, *pluginObj);
    return queryballpointlayer;
}

/*********************************************************
 *
 *               Load Weights
 ********************************************************/
float ieee_float(uint32_t f)
{
    std::cout<<f<<"  ";
    return *((float*)((void*)(&f)));
}
std::map<std::string, Weights> loadWeights(const std::string file)
{
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;
    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");
    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");
    while (count--)
    {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;
        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;

        wt.type = DataType::kFLOAT;
        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];

            //std::cout<<ieee_float(val[x])<<std::endl;
        }
        wt.values = val;
        wt.count = size;
        weightMap[name] = wt;
    }
    return weightMap;
}

IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
    float *gamma = (float*)weightMap[lname + ".weight"].values;
    float *beta = (float*)weightMap[lname + ".bias"].values;
    float *mean = (float*)weightMap[lname + ".running_mean"].values;
    float *var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;

    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};

    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{DataType::kFLOAT, pval, len};
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

/************************************************************************************
 *
 *             Conv->BN->Leaky
 *
 *************************************************************************************/
ILayer* convBnLeaky(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input,  int outch, int ksize, int s, int p, std::string key_conv, std::string key_bn)
{

    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ksize, ksize}, weightMap[key_conv + ".weight"],  weightMap[key_conv + ".bias"]);
    assert(conv1);
    conv1->setStrideNd(DimsHW{s, s});
    conv1->setPaddingNd(DimsHW{p, p});
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), key_bn, 1e-5);
    auto lr = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    return lr;
}

/************************************************************************************
 *
 *             farthest_point_sample(xyz, S)
 *
 *************************************************************************************/
ILayer* farthest_point_sample(INetworkDefinition *network,ITensor** xyz,int num, int S)
{
     auto creator = getPluginRegistry()->getPluginCreator("FursthestPointSample_TRT", "1");
     std::vector<nvinfer1::PluginField>f;
     nvinfer1::PluginFieldCollection fc;
     f.emplace_back(PluginField("num", &num, PluginFieldType::kINT8, 1));
     f.emplace_back(PluginField("nSamples",&S, PluginFieldType::kINT8, 1));

     fc.nbFields = f.size();
     fc.fields = f.data();
     IPluginV2 *pluginObj = creator->createPlugin("furthestsamplelayer", &fc);
     auto farthest_point_sample_layer = network->addPluginV2(xyz, 1, *pluginObj);
     return farthest_point_sample_layer;
}


ILayer* group_point(INetworkDefinition *network, ITensor** inputs,int n, int c,int npoints, int nsamples)
{
    auto creator                          = getPluginRegistry()->getPluginCreator("GroupPointPlugin_TRT", "1");
      std::vector<nvinfer1::PluginField>f;
      nvinfer1::PluginFieldCollection fc;
      f.emplace_back(PluginField("N",  &n,   PluginFieldType::kINT8, 1));
      f.emplace_back(PluginField("C",         &c,  PluginFieldType::kINT8, 1));
      f.emplace_back(PluginField("npoints",    &npoints,   PluginFieldType::kINT8, 1));
      f.emplace_back(PluginField("nsamples",         &nsamples,    PluginFieldType::kINT8, 1));

      fc.nbFields = f.size();
      fc.fields   = f.data();
      IPluginV2 *pluginObj = creator->createPlugin("grouppointlayer", &fc);
      auto grouppointlayer = network->addPluginV2(inputs, 2, *pluginObj);
      return grouppointlayer;
}



std::vector<ILayer*> PointNetSetAbstractionMsg(INetworkDefinition *network, ITensor** inputs, int npoint, std::vector<float>radius_list, std::vector<int>nsample_list, std::vector<std::vector<int>>mlp_list, std::map<std::string, Weights>& weightMap,std::string sa_name)
{
      std::vector<ILayer*> new_points_list;
      for(int i=0; i<3; i++)
      {
          int nsamples = nsample_list[0];
          int n = inputs[0]->getDimensions().d[1];
          int c_xyz = inputs[0]->getDimensions().d[2];
          int npoints = inputs[1]->getDimensions().d[1];

          auto group_idx = query_ball_point(network, inputs, n, npoints, nsamples, radius_list[0]); // query_input

          ITensor* group_xyz_input[] = {inputs[0], group_idx->getOutput(0)};// [xyz, idx]
          auto group_xyz_layer = group_point(network, group_xyz_input, n,  c_xyz, npoints,  nsamples);
          auto group_xyz_shuffle = network->addShuffle(*group_xyz_layer->getOutput(0));
          group_xyz_shuffle->setFirstTranspose(Permutation{0,2,3,1});
          auto new_xyz_view = network->addShuffle(*inputs[1]);
          new_xyz_view->setReshapeDimensions(Dims4{inputs[1]->getDimensions().d[0],inputs[1]->getDimensions().d[1],1,inputs[1]->getDimensions().d[2]});
          auto grouped_xyz = network->addElementWise(*group_xyz_shuffle->getOutput(0), *new_xyz_view->getOutput(0), ElementWiseOperation::kSUB);
          auto grouped_xyz_3_512_32 = network->addShuffle(*grouped_xyz->getOutput(0));
          grouped_xyz_3_512_32->setFirstTranspose(Permutation{0,3,1,2});

          ITensor *group_point_input[] = {inputs[2], group_idx->getOutput(0)};
          int points_channels = inputs[2]->getDimensions().d[2];

          auto group_points_layer = group_point(network,group_point_input, n, points_channels, npoints,nsamples);

          ITensor* grouped_points_xyz_input [] = { group_points_layer->getOutput(0), grouped_xyz_3_512_32->getOutput(0)};
          auto grouped_points_cat = network->addConcatenation(grouped_points_xyz_input,2);  // 6*512*32
          auto grouped_points_shuffle = network->addShuffle(*grouped_points_cat->getOutput(0));
          grouped_points_shuffle->setFirstTranspose(Permutation{0,1,3,2});
          auto grouped_points = grouped_points_shuffle->getOutput(0);  // 1 6,32,512

          ILayer* grouped_points_layer;
          for(int j=0; j<mlp_list[i].size(); j++)
          {
             std::string conv_block_name = sa_name+".conv_blocks."+std::to_string(i)+ "." + std::to_string(j);
             std::string bn_block_name   = sa_name+".bn_blocks."  +std::to_string(i)+ "." + std::to_string(j);
             grouped_points_layer = convBnLeaky(network,weightMap,*grouped_points, mlp_list[i][j], 1, 1, 0, conv_block_name, bn_block_name);
             grouped_points = grouped_points_layer->getOutput(0);
         }

         auto new_points =network->addTopK(*grouped_points_layer->getOutput(0),TopKOperation::kMAX,1,0x04);
         auto new_points_shuffle = network->addShuffle(*new_points->getOutput(0));
         new_points_shuffle->setReshapeDimensions(Dims3(new_points->getOutput(0)->getDimensions().d[1],
                                                 new_points->getOutput(0)->getDimensions().d[2],
                                                 new_points->getOutput(0)->getDimensions().d[3]));

         new_points_list.push_back(new_points_shuffle);

    } // for
    return new_points_list;
}



std::vector<ILayer*> PointNetSetAbstractionMsg2(INetworkDefinition *network, ITensor** inputs, int npoint, std::vector<float>radius_list, std::vector<int>nsample_list, std::vector<std::vector<int>>mlp_list, std::map<std::string, Weights>& weightMap,std::string sa_name)
{

      std::vector<ILayer*> new_points_list;
      auto xyz = inputs[0];
      auto new_xyz = inputs[1];
      auto points = inputs[2];
      for(int i=0; i<radius_list.size(); i++)
      {
          int nsamples = nsample_list[0];
          int n       = xyz->getDimensions().d[1];
          int c_xyz   = xyz->getDimensions().d[2];
          int npoints = new_xyz->getDimensions().d[1];
          ITensor* query_ball_input[] = {xyz, new_xyz};
          auto group_idx = query_ball_point(network, query_ball_input, n, npoints, nsamples, radius_list[0]); // group_idx test pass

          ITensor* group_xyz_input[]   = {xyz, group_idx->getOutput(0)};// [xyz, idx]
          ITensor *group_point_input[] = {points, group_idx->getOutput(0)};
          int points_channels = points->getDimensions().d[2];
          auto group_points_layer = group_point(network,group_point_input, n, points_channels, npoints,nsamples);  // grouped_points test pass
          auto group_xyz_layer    = group_point(network, group_xyz_input, n, c_xyz, npoints,nsamples);


          auto group_xyz_shuffle = network->addShuffle(*group_xyz_layer->getOutput(0));
          group_xyz_shuffle->setFirstTranspose(Permutation{0,2,3,1});

          auto new_xyz_view = network->addShuffle(*new_xyz);
          new_xyz_view->setReshapeDimensions(Dims4{new_xyz->getDimensions().d[0],new_xyz->getDimensions().d[1],1,new_xyz->getDimensions().d[2]});

          auto grouped_xyz_sub = network->addElementWise(*group_xyz_shuffle->getOutput(0), *new_xyz_view->getOutput(0), ElementWiseOperation::kSUB);
          grouped_xyz_sub->getOutput(0)->setName((sa_name + std::to_string(i)).c_str());
          auto grouped_xyz_3_512_32 = network->addShuffle(*grouped_xyz_sub->getOutput(0));
          grouped_xyz_3_512_32->setFirstTranspose(Permutation{0,3,1,2});


          ITensor* grouped_points_xyz_input [] = { group_points_layer->getOutput(0), grouped_xyz_3_512_32->getOutput(0)};

          auto grouped_points_cat = network->addConcatenation(grouped_points_xyz_input,2);  // 1 323 128 64   test pas

          auto grouped_points_cat_shuffle = network->addShuffle(*grouped_points_cat->getOutput(0));
          grouped_points_cat_shuffle->setFirstTranspose(Permutation{0,1,3,2});

          auto grouped_points = grouped_points_cat_shuffle->getOutput(0);  // 1 6,32,512

          ILayer* grouped_points_layer;
          for(int j=0; j<mlp_list[i].size(); j++)
          {

                std::string conv_block_name = sa_name+".conv_blocks."+std::to_string(i)+ "." + std::to_string(j);
                std::string bn_block_name   = sa_name+".bn_blocks."  +std::to_string(i)+ "." + std::to_string(j);

                grouped_points_layer = convBnLeaky(network,weightMap,*grouped_points, mlp_list[i][j], 1, 1, 0, conv_block_name, bn_block_name);
                grouped_points = grouped_points_layer->getOutput(0);

         }

       auto new_points =network->addTopK(*grouped_points_layer->getOutput(0),TopKOperation::kMAX,1,0x04);

       auto new_points_shuffle = network->addShuffle(*new_points->getOutput(0));

       new_points_shuffle->setReshapeDimensions(Dims3(new_points->getOutput(0)->getDimensions().d[1],
                                                new_points->getOutput(0)->getDimensions().d[2],
                                                new_points->getOutput(0)->getDimensions().d[3]));
       //LogTensor(new_points_shuffle->getOutput(0),sa_name);
       new_points_list.push_back(new_points_shuffle);

   } // for

   return new_points_list;  // test pass
}

ILayer* PointNetSetAbstraction(INetworkDefinition *network, ITensor** inputs,std::vector<int>mlp_list, std::map<std::string, Weights>& weightMap,std::string sa_name)
{

    //sample and group all
    auto xyz = network->addShuffle(*inputs[0]);
    xyz->setFirstTranspose(Permutation{2,0,1});
    ITensor* input[] ={xyz->getOutput(0), inputs[1]};
    auto new_points_cat = network->addConcatenation(input,2) ;

    auto new_points_shuffle = network->addShuffle(*new_points_cat->getOutput(0));
    new_points_shuffle->setReshapeDimensions(Dims4{1,new_points_cat->getOutput(0)->getDimensions().d[0],
                                           new_points_cat->getOutput(0)->getDimensions().d[2],
                                             new_points_cat->getOutput(0)->getDimensions().d[1]});


    auto new_points = new_points_shuffle->getOutput(0);

    ILayer* grouped_points_layer;

    for(int j=0; j<mlp_list.size(); j++)
    {

        std::string conv_block_name = sa_name+".mlp_convs."+ std::to_string(j);
        std::string bn_block_name   = sa_name+".mlp_bns."   + std::to_string(j);

        grouped_points_layer = convBnLeaky(network,weightMap,*new_points, mlp_list[j], 1, 1, 0, conv_block_name, bn_block_name);
        new_points = grouped_points_layer->getOutput(0);


    }
    auto new_points_output = network->addTopK(*grouped_points_layer->getOutput(0),TopKOperation::kMAX,1,0x04);
    auto new_points_output_shuffle = network->addShuffle(*new_points_output->getOutput(0));
    new_points_output_shuffle->setReshapeDimensions(Dims3{new_points_output->getOutput(0)->getDimensions().d[0],
                                          new_points_output->getOutput(0)->getDimensions().d[1],
                                            new_points_output->getOutput(0)->getDimensions().d[2]});
    return new_points_output_shuffle;
}


IPluginV2Layer* featureInterpolate(INetworkDefinition* network, ITensor** inputs, int N, int M, int Ch)
{
    auto creator = getPluginRegistry()->getPluginCreator("FeatureInterpolatedPlugin_TRT", "1");
    std::vector<nvinfer1::PluginField>f;
    nvinfer1::PluginFieldCollection fc;

    f.emplace_back(PluginField("N",  &N,     PluginFieldType::kINT8, 1));
    f.emplace_back(PluginField("M",  &M,     PluginFieldType::kINT8, 1));
    f.emplace_back(PluginField("C",  &Ch,     PluginFieldType::kINT8, 1));

    fc.nbFields = f.size();
    fc.fields   = f.data();

    IPluginV2* pluginObj = creator->createPlugin("featureInterpolate", &fc);
    auto feature_interpolate_layer =network->addPluginV2(inputs, 3 ,*pluginObj);
    return feature_interpolate_layer;
}


ILayer* FP3(INetworkDefinition* network, ITensor** inputs, std::vector<int>mlp_list,std::map<std::string, Weights>& weightMap, std::string fp_name)
{

    auto points2  = network->addShuffle(*inputs[2]); // points2
    points2->setFirstTranspose(Permutation{2,1,0});
    std::vector<ITensor*>repeat_points(128, points2->getOutput(0));
    auto interpolated_points_cat = network->addConcatenation(&repeat_points[0],128);//128,1,1024

    auto  interpolated_points_cat_shuffle = network->addShuffle(*interpolated_points_cat->getOutput(0)); // points2
    interpolated_points_cat_shuffle->setFirstTranspose(Permutation{1,2,0});


    ITensor*input[] = {inputs[1], interpolated_points_cat_shuffle->getOutput(0)};
    auto new_points_cat = network->addConcatenation(input,2);//128,1,1024


    auto new_points = new_points_cat->getOutput(0);
    ILayer* fp3;
    for(int j=0; j<mlp_list.size(); j++)
    {
         std::string conv_block_name = fp_name+".mlp_convs."+ std::to_string(j);
         std::string bn_block_name   = fp_name+".mlp_bns."   + std::to_string(j);
         fp3  = convBnLeaky(network,weightMap,*new_points, mlp_list[j], 1, 1, 0, conv_block_name, bn_block_name);
         new_points = fp3->getOutput(0);
    }
    return fp3;
}


ILayer* FP2(INetworkDefinition* network, ITensor** inputs, std::vector<int>mlp_list, std::map<std::string, Weights> weightMap, std::string fp_name)
{
    auto new_points = network->addShuffle(*inputs[3]);
    new_points->setFirstTranspose(Permutation{1,0,2});

    ITensor* input[] = {inputs[0], inputs[1], new_points->getOutput(0)};
    auto feature = featureInterpolate(network, input, inputs[0]->getDimensions().d[1],inputs[1]->getDimensions().d[1], new_points->getOutput(0)->getDimensions().d[1]);
    auto points1 = network->addShuffle(*inputs[2]);
    points1->setFirstTranspose(Permutation{2,0,1});
    auto points2 = network->addShuffle(*feature->getOutput(0));
    points2->setFirstTranspose(Permutation{1,0,2});
    ITensor*input_tmp[] = {points1->getOutput(0),points2->getOutput(0)};
    auto interpolated_points_cat = network->addConcatenation(input_tmp,2)->getOutput(0);

    ILayer* fp2;
    for(int j=0; j<mlp_list.size(); j++)
    {
        std::string conv_block_name = fp_name+".mlp_convs."+ std::to_string(j);
        std::string bn_block_name   = fp_name+".mlp_bns."   + std::to_string(j);
        fp2  = convBnLeaky(network,weightMap,*interpolated_points_cat, mlp_list[j], 1, 1, 0, conv_block_name, bn_block_name);
        interpolated_points_cat = fp2->getOutput(0);
    }

    return fp2;
}


ILayer* FP1(INetworkDefinition* network, ITensor** inputs, std::vector<int>mlp_list, std::map<std::string, Weights> weightMap, std::string fp_name)
{
    auto new_points = network->addShuffle(*inputs[2]);
    new_points->setFirstTranspose(Permutation{1,0,2});
    ITensor* input[] = {inputs[0], inputs[1],new_points->getOutput(0)};
    auto feature = featureInterpolate(network, input, inputs[0]->getDimensions().d[1],inputs[1]->getDimensions().d[1], new_points->getOutput(0)->getDimensions().d[1])->getOutput(0);  
    auto interpolated_points_cat = network->addShuffle(*feature);
    interpolated_points_cat->setFirstTranspose(Permutation{1,0,2});
    auto conv_points = interpolated_points_cat->getOutput(0);
    ILayer* fp1;
    for(int j=0; j<mlp_list.size(); j++)
    {
        std::string conv_block_name = fp_name+".mlp_convs."+ std::to_string(j);
        std::string bn_block_name   = fp_name+".mlp_bns."   + std::to_string(j);    
        fp1  = convBnLeaky(network,weightMap,*conv_points, mlp_list[j], 1, 1, 0, conv_block_name, bn_block_name);
        conv_points = fp1->getOutput(0);
    }

    return fp1;
}

#define TRT 0
ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt)
{
    std::map<std::string, Weights> weightMap = loadWeights("pointnet.wts");

    INetworkDefinition* network = builder->createNetworkV2(0U);

    ITensor* xyz        = network->addInput(INPUT_BLOB_NAME,dt, Dims3{1,8000,C});


    auto new_xyz_512 = farthest_point_sample(network,&xyz,8000, 512);

    auto new_xyz = new_xyz_512->getOutput(0);
    ITensor* query_input[] = {xyz, new_xyz, xyz};
    int S1 = 512;
    std::vector<float>radius_list_sa1{0.1, 0.2,0.3};
    std::vector<int>nsample_list_sa1{32, 64, 128};
    std::vector<std::vector<int>>mlp_sa1 = {{32, 32, 64}, {64, 64, 128}, {64, 96, 128}};
    auto sa1 = PointNetSetAbstractionMsg(network,query_input,S1,radius_list_sa1, nsample_list_sa1, mlp_sa1,weightMap,"sa1");
    ITensor* new_points_input_sa1 [] = { sa1[0]->getOutput(0), sa1[1]->getOutput(0), sa1[2]->getOutput(0)};
    auto new_points_cat_sa1 = network->addConcatenation(new_points_input_sa1,3);
    auto l1_points = network->addShuffle(*new_points_cat_sa1->getOutput(0));
    l1_points->setFirstTranspose(Permutation{1,2,0}); //  pass


    ///////////////////////////////////// sa2  //////////////////////////////////////////////
    int S2 = 128;
    auto new_xyz_128_1 = farthest_point_sample(network,&new_xyz,512, S2);
    auto new_xyz_128 = new_xyz_128_1->getOutput(0);
    std::vector<float>radius_list_sa2{0.3, 0.6};
    std::vector<int>nsample_list_sa2{ 64, 128};
    std::vector<std::vector<int>>mlp_sa2 = {{128,128,256}, {128,196,256}};
    ITensor* query_input_sa2[] = {new_xyz, new_xyz_128, l1_points->getOutput(0)};
    auto sa2 = PointNetSetAbstractionMsg2(network,query_input_sa2,S2,radius_list_sa2,nsample_list_sa2,mlp_sa2,weightMap,"sa2");
    ITensor* new_points_input_sa2 [] = { sa2[0]->getOutput(0), sa2[1]->getOutput(0)};
    auto l2_points = network->addConcatenation(new_points_input_sa2,2);

    ///////////////////////////////////// sa3  //////////////////////////////////////////////
    ITensor*input3[] = {new_xyz_128, l2_points->getOutput(0)};
    std::vector<int>mlp_sa3 = {256, 512,1024};
    auto l3_points = PointNetSetAbstraction(network,input3,mlp_sa3, weightMap,"sa3");// pass
   ///////////////////////////////////// FP3  //////////////////////////////////////////////

    ITensor* fp3_input[] = {new_xyz_128,l2_points->getOutput(0), l3_points->getOutput(0)};
    std::vector<int>mlp_fp3 = {256, 256};
    auto fp3 = FP3(network,fp3_input,mlp_fp3,weightMap,"fp3");// pass

    ///////////////////////////////////// FP2  //////////////////////////////////////////////
    ITensor* fp2_input[]= {new_xyz, new_xyz_128, l1_points->getOutput(0), fp3->getOutput(0)};
    std::vector<int>mlp_fp2= {256, 128};
    auto fp2 = FP2(network, fp2_input, mlp_fp2, weightMap, "fp2");

   ///////////////////////////////////// FP1  //////////////////////////////////////////////

    ITensor* fp1_input[] = {xyz, new_xyz,fp2->getOutput(0)};
    std::vector<int>mlp_fp1= {128, 128};
    auto fp1 = FP1(network, fp1_input, mlp_fp1, weightMap, "fp1");
    auto feat = convBnLeaky(network, weightMap, *fp1->getOutput(0),128,1,1,0,"conv1","bn1");
    IConvolutionLayer* conv2 = network->addConvolutionNd(*feat->getOutput(0), 2, DimsHW{1, 1}, weightMap["conv2.weight"],  weightMap["conv2.bias"]);
    assert(conv2);
    conv2->setStrideNd(DimsHW{1, 1});
    auto seg = network->addShuffle(*conv2->getOutput(0));
    seg->setFirstTranspose(Permutation{1,0,2});
    LogTensor(conv2->getOutput(0), "seg");

    ///////////////////////////////////// OUTPUT //////////////////////////////////////////////
    seg->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(* seg->getOutput(0)); //256*512

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    if(engine!=nullptr)
       std::cout << "Build engine successfully!" << std::endl;

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


int main()
{
   QFile  file_pt("Trailer_1.csv");
   float xyz  [1*8000*3];
    float meax=0,meay=0,meaz=0,scale=0;
   if(file_pt.open(QIODevice::ReadOnly))
   {
        QTextStream in_pt(&file_pt);
        int i=0;
        while(!in_pt.atEnd())
        {
             QString line = in_pt.readLine();
             QStringList strlist = line.split(",");
             xyz[i*3+0]  = strlist[0].toFloat();
             xyz[i*3+1]  = strlist[1].toFloat();
             xyz[i*3+2]  = strlist[2].toFloat();

             meax+=xyz[i*3+0];
             meay+=xyz[i*3+1];
             meaz+=xyz[i*3+2];
             i++;
        }
    }

            meax/=8000; meay/=8000; meaz/=8000;
            //std::cout<<meax<<":"<<meay<<" "<<meaz<<std::endl;
            // substract mean and divide scale
            for(int i=0; i<8000; i++)
            {
               float x = xyz[i*3+0]-meax;
               float y = xyz[i*3+1]-meay;
               float z = xyz[i*3+2]-meaz;

              xyz[i*3+0]=x;
              xyz[i*3+1]=y;
              xyz[i*3+2]=z;

               if(sqrt(x*x+y*y+z*z)>scale)
               {
                   scale = sqrt(x*x+y*y+z*z);
               }

         }

            for(int i=0; i<8000; i++)
            {
               float x = xyz[i*3+0]/scale;
               float y = xyz[i*3+1]/scale;
               float z = xyz[i*3+2]/scale;

              xyz[i*3+0]=x;
              xyz[i*3+1]=y;
              xyz[i*3+2]=z;

             }



    char *trtModelStream{nullptr};
    size_t size{0};

    IHostMemory* modelStream{nullptr};
#if TRT
    APIToModel(1, &modelStream);
    assert(modelStream != nullptr);
     std::ofstream p("pointnet2.engine", std::ios::binary);
     if (!p) {
              std::cerr << "could not open plan output file" << std::endl;
              return -1;
          }
      p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
      modelStream->destroy();
#else
    std::ifstream file("pointnet2.engine", std::ios::binary);
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

     std::cout<<"-- "<<(engine->getNbBindings())<<std::endl;
     void* buffers[2];

     const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
     const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
     float *output= new float[1*2*8000];
     auto start_read = std::chrono::system_clock::now();

      cudaMalloc((void**)&buffers[inputIndex], 1 * 3 *8000 * sizeof(float));
      cudaMalloc((void**)&buffers[outputIndex],1*2*8000 * sizeof(float));

     // Create stream
     cudaStream_t stream;
     cudaStreamCreate(&stream);
      // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
     cudaMemcpyAsync(buffers[inputIndex], xyz, 1 * 3 * 8000 * sizeof(float), cudaMemcpyHostToDevice, stream);
     context->enqueue(1, buffers, stream, nullptr);
     cudaMemcpyAsync(output, buffers[outputIndex], 1*2*8000 * sizeof(float), cudaMemcpyDeviceToHost, stream);
     cudaStreamSynchronize(stream);
     auto end_read = std::chrono::system_clock::now();
     double during_time_read = std::chrono::duration_cast<std::chrono::milliseconds>(end_read - start_read).count();
     std::cout <<"time consume during_time_read===" <<  during_time_read << "ms" << std::endl;
     int csv_file_fd   = open("Trailer.csv",O_WRONLY | O_CREAT | O_TRUNC, 0644);
     FILE* csv_file_g   = fdopen(csv_file_fd, "w");
     for(int i=0; i<8000;i++)
      {   //std::cout<<output[0*8000+i]<<":"<<output[1*8000+i]<<std::endl;
            if(output[0*8000+i]>output[1*8000+i])
            {
                fprintf(csv_file_g,"%f,%f,%f,%d\n", xyz[i*3+0],xyz[i*3+1],xyz[i*3+2],255);
            }
            else
                 fprintf(csv_file_g,"%f,%f,%f,%d\n", xyz[i*3+0],xyz[i*3+1],xyz[i*3+2],120);
       }
      fclose(csv_file_g);

     cudaStreamDestroy(stream);
     cudaFree(buffers[inputIndex]);
     cudaFree(buffers[outputIndex]);
     context->destroy();
     engine->destroy();
     runtime->destroy();

#endif
    return 0;
}
