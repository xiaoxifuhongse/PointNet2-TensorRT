#include <fstream>
#include <fcntl.h>
#include "NvInfer.h"
#include "logging.h"
#include "cuda_utils.h"
#define NUM (8000)
using namespace nvinfer1;
static Logger gLogger;

class WheelDetTRT
{
public: 
	 WheelDetTRT();
	 ~WheelDetTRT();
     bool init(std::string trt_file);
     void forward(std::vector<inno_point>pts);
private:
    void prepareData(std::vector<inno_point>pts, float*xyz, float& meax, float& meay, float& meaz, float& scale);
	ICudaEngine* engine ;
    IExecutionContext* context;
    cudaStream_t stream;
    void  *buffers[2];
    float *output;
    float xyz  [1*NUM*3]= {0.f};
};
