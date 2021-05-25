#include "wheelDet_trt.h"
#include <set>
#include<string.h>
WheelDetTRT::WheelDetTRT()
{

}

WheelDetTRT::~WheelDetTRT()
{
	delete []output;
 	cudaStreamDestroy(stream);
	cudaFree(buffers[0]);
	cudaFree(buffers[1]);
	context->destroy();
	engine->destroy();
}

bool WheelDetTRT::init(std::string trt_file)
{
	char *trtModelStream{nullptr};
    size_t size{0};
	std::ifstream file(trt_file, std::ios::binary);
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
	else
	{
		return false;
	}

	IRuntime* runtime = createInferRuntime(gLogger);
	if(runtime == nullptr) return false;
	ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
	if(engine == nullptr) return false;
	IExecutionContext* context = engine->createExecutionContext();
	if(context == nullptr) return false;
	delete[] trtModelStream;
	runtime->destroy();
    output= new float[1*2*NUM];
    //auto start_read = std::chrono::system_clock::now();

    cudaMalloc((void**)&buffers[0], 1 * 3 * NUM * sizeof(float));
    cudaMalloc((void**)&buffers[1], 1 * 2 * NUM * sizeof(float));
	cudaStreamCreate(&stream);
    return true;
}


void WheelDetTRT::forward(std::vector<inno_point>pts)
{
	float meax  = 0.0f;
	float meay  = 0.0f;
    float meaz  = 0.0f;
    float scale = 0.0f;
    prepareData(pts, xyz, meax, meay, meaz, scale);
    cudaMemcpyAsync(buffers[0], xyz, 1 * 3 * NUM * sizeof(float), cudaMemcpyHostToDevice, stream);
    context->enqueue(1, buffers, stream, nullptr);
    cudaMemcpyAsync(output, buffers[1], 1*2*NUM * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

}


void WheelDetTRT::prepareData(std::vector<inno_point> pts, float *xyz, float &meax, float &meay, float &meaz, float &scale)
{
    int size = pts.size();
    memset(xyz, 0.0, sizeof(float)*1*3*NUM);
    if(size<=NUM)
    {
        float meax=0.f;
        float meay=0.f;
        float meaz=0.f;
        for(int i=0; i<size;i++)
        {
          float x = pts[i].x;
          float y = pts[i].y;
          float z = pts[i].z;
          meax+=x;
          meay+=y;
          meaz+=z;
        } // get mean

        meax/=size; meay/=size; meaz/=size;
        for(int i=0; i<size; i++)
        {
            float x = pts[i].x-meax;
            float y = pts[i].y-meay;
            float z = pts[i].z-meaz;
            xyz[i*3+0] = x;
            xyz[i*3+1] = y;
            xyz[i*3+2] = z;
            if(sqrt(x*x+y*y+z*z)>scale)
            {
                scale = sqrt(x*x+y*y+z*z);
            }
        } // get scale

        for(int i=0; i<size; i++)
        {

           xyz[i*3+0] /= scale;
           xyz[i*3+1] /= scale ;
           xyz[i*3+2] /= scale;
         }
    } // if
    else
    {
        meax = 0.0f;
        meay = 0.0f;
        meaz = 0.0f;
        scale= 0.0f;
        std::set<int> s;
        srand( (unsigned)time(NULL) );
        while(s.size()<=NUM)
        {
            s.insert(rand()%size);
        }
        std::vector<inno_point>choice;
        for (std::set<int>::iterator it = s.begin(); it != s.end(); ++it)
        {
           inno_point pt;
           pt.x = pts[*it].x;
           pt.y = pts[*it].y;
           pt.z = pts[*it].z;
           choice.push_back(pt);
           meax+=pt.x;
           meay+=pt.y;
           meaz+=pt.z;
        }
        meax = meax/NUM;
        meay = meay/NUM;
        meaz = meaz/NUM;

        for(int i=0; i<choice.size(); i++)
        {
            float x= choice[i].x-meax;
            float y= choice[i].y-meay;
            float z= choice[i].z-meaz;
            float dist = sqrt(x*x+y*y+z*z);
            xyz[i*3+0] = x;
            xyz[i*3+1] = y;
            xyz[i*3+2] = z;

            if(dist>scale)
            {
                 scale = dist;
            }
        }

        for(int i=0; i<NUM; i++)
        {

           xyz[i*3+0] /= scale;
           xyz[i*3+1] /= scale ;
           xyz[i*3+2] /= scale;
         }

    }// else
}
