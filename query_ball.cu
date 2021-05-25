#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"
extern "C"

void run(std::vector<inno_point>&pts);

__device__ void __update(float *__restrict__ dists, int *__restrict__ dists_i,int idx1, int idx2)
{
  const float v1 = dists[idx1], v2 = dists[idx2];
  const int i1 = dists_i[idx1], i2 = dists_i[idx2];
  dists[idx1] = max(v1, v2);
  dists_i[idx1] = v2 > v1 ? i2 : i1;
}

// Input dataset: (b, n, 3), tmp: (b, n)
// Ouput idxs (b, m)
template <unsigned int block_size>
__global__ void furthest_point_sampling_kernel(
    int b, int n, int m, const float *__restrict__ dataset,
    float *__restrict__ temp, int *__restrict__ idxs) {
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
  for (int j = 1; j < m; j++) {
    int besti = 0;
    float best = -1;
    float x1 = dataset[old * 3 + 0];
    float y1 = dataset[old * 3 + 1];
    float z1 = dataset[old * 3 + 2];
    for (int k = tid; k < n; k += stride) {
      float x2, y2, z2;
      x2 = dataset[k * 3 + 0];
      y2 = dataset[k * 3 + 1];
      z2 = dataset[k * 3 + 2];
     // printf("%f,%f,%f\n",x2,y2,z2);
      float mag = (x2 * x2) + (y2 * y2) + (z2 * z2);
      if (mag <= 1e-3) continue;

      float d =
          (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);

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
                                             const float *dataset, float *temp,
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
            out[c*j+l] = points[c*a + l];
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




// input: points(b, c, n) idx(b, npoints, nsample)
// output: out(b, c, npoints, nsample)
__global__ void group_points_kernel(int b, int c, int n, int npoints,
                                    int nsample,
                                    const float *__restrict__ points,
                                    const int *__restrict__ idx,
                                    float *__restrict__ out) {




  int batch_index = blockIdx.x;
  points += batch_index * n * c;
  idx += batch_index * npoints * nsample;
  out += batch_index * npoints * nsample * c;

  const int index = threadIdx.y * blockDim.x + threadIdx.x;
  const int stride = blockDim.y * blockDim.x;


  for (int i = index; i < c * npoints; i += stride) {
    const int l = i / npoints;
    const int j = i % npoints;

    for (int k = 0; k < nsample; ++k) {
      int ii = idx[j * nsample + k];
       // printf("%f\n",points[ii * 3 + l]);
      out[(l * npoints + j) * nsample + k] = points[ii * 3 + l];
    }
  }
}

void group_points_kernel_wrapper(int b, int c, int n, int npoints, int nsample,
                                 const float *points, const int *idx,
                                 float *out) {


    group_points_kernel<<<b, opt_block_config(npoints, c),0>>>(b, c, n, npoints, nsample, points, idx, out);

  CUDA_CHECK_ERRORS();
}






// input: new_xyz(b, m, 3) xyz(b, n, 3)
// output: idx(b, m, nsample)
__global__ void query_ball_point_kernel(int b, int n, int m, float radius,
                                        int nsample,
                                        const float *__restrict__ new_xyz,
                                        const float *__restrict__ xyz,
                                        int *__restrict__ idx) {



  int batch_index = blockIdx.x;
  xyz += batch_index * n * 3;
  new_xyz += batch_index * m * 3;
  idx += m * nsample * batch_index;

  int index = threadIdx.x;
  int stride = blockDim.x;

  float radius2 = radius * radius;
  for (int j = index; j < m; j += stride) {
    float new_x = new_xyz[j * 3 + 0];
    float new_y = new_xyz[j * 3 + 1];
    float new_z = new_xyz[j * 3 + 2];
    for (int k = 0, cnt = 0; k < n && cnt < nsample; ++k) {
      float x = xyz[k * 3 + 0];
      float y = xyz[k * 3 + 1];
      float z = xyz[k * 3 + 2];
      float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) +
                 (new_z - z) * (new_z - z);
      if (d2 < radius2) {
        if (cnt == 0) {
          for (int l = 0; l < nsample; ++l) {
            idx[j * nsample + l] = k;
          }
        }
        idx[j * nsample + cnt] = k;
        ++cnt;
      }
    }
  }
}

void query_ball_point_kernel_wrapper(int b, int n, int m, float radius,
                                     int nsample, const float *new_xyz,
                                     const float *xyz, int *idx) {

  query_ball_point_kernel<<<b, opt_n_threads(m), 0>>>(b, n, m, radius, nsample, new_xyz, xyz, idx);//(512,1)

  CUDA_CHECK_ERRORS();
}
#include <fcntl.h>
void run(std::vector<inno_point> &pts)
{
    int B = 1;
    int N = 8000;
    int Samples = 128;
    float radius = 0.4;
    int K = 128;
    // input point data host
    float* h_points = new float[3*N];
    for(int i=0; i<N;i++)
    {
        h_points[i*3+0]=pts[i].x;
        h_points[i*3+1]=pts[i].y;
        h_points[i*3+2]=pts[i].z;
    }
    // point in device
    float *d_points;
    cudaMalloc((void**)&d_points, 3*N*sizeof(float)) ;
    cudaMemcpy(d_points,h_points,3*N*sizeof(float),cudaMemcpyHostToDevice);
    // temp for furthest sample points
    float *d_temp;
    cudaMalloc((void**)&d_temp,N*sizeof(float));
    cudaMemset(d_temp, 1e10,N*sizeof(float));
    // output index
    int *d_index = new int[Samples];
    int *h_index = new int[Samples];

    cudaMalloc((void**)&d_index, Samples*sizeof(int)) ;
    cudaMemset(d_index,0,Samples*sizeof(int));

    furthest_point_sampling_kernel_wrapper(B, N,Samples, d_points, d_temp, d_index);
    cudaMemcpy(h_index,d_index,Samples*sizeof(int),cudaMemcpyDeviceToHost);

    for(int i=0; i<Samples; i++)
               printf("i=%d, index=%d\n",i, h_index[i] );
    float * d_points_sample ;
    float * h_point_sampled = new float[3*Samples];
    cudaMalloc((void**)&d_points_sample, 3*Samples*sizeof(float));
    cudaMemset(d_points_sample,0,3*Samples*sizeof(int));
    gather_points_kernel_wrapper(B,3,N,Samples,d_points,d_index,d_points_sample);
    cudaMemcpy(h_point_sampled,d_points_sample,3*Samples*sizeof(float),cudaMemcpyDeviceToHost);



    int *d_queryidx;
    int*h_queryidx = new int[Samples*K];
    cudaMalloc((void**)&d_queryidx,Samples*K*sizeof(int));

    query_ball_point_kernel_wrapper(B,N,Samples,radius,K,d_points_sample,d_points, d_queryidx);
    cudaMemcpy(h_queryidx,d_queryidx,Samples*K*sizeof(int),cudaMemcpyDeviceToHost);

//    int csv_file_fd   = open("test2.csv",O_WRONLY | O_CREAT | O_TRUNC, 0644);
//    FILE* csv_file_g   = fdopen(csv_file_fd, "w");
//    for(int i=0; i<Samples;i++)
//    {
//        for(int j=0; j<K;j++){
//            int n = h_queryidx[i*32+j];
//          fprintf(csv_file_g,"%f,%f,%f,%d\n",pts[n].x,pts[n].y,pts[n].z, 255);
//        }
//    }

//    fclose(csv_file_g);

//     for(int i=0; i<Samples; i++)
//     {   printf("%d: ",i);
//         for(int j=0; j<K; j++)
//             printf("%d  ", h_queryidx[i*32+j]);
//          printf("\n");
//     }

     float * d_output ;
     float * h_output= new float[3*Samples*K];
     cudaMalloc((void**)&d_output, sizeof(float)*Samples*K*3);

     group_points_kernel_wrapper(B,3,N,Samples,K,d_points,d_queryidx,d_output);
     cudaMemcpy(h_output,d_output,3*Samples*K*sizeof(float),cudaMemcpyDeviceToHost);
         int csv_file_fd   = open("test2.csv",O_WRONLY | O_CREAT | O_TRUNC, 0644);
         FILE* csv_file_g   = fdopen(csv_file_fd, "w");
         for(int i=0; i<Samples;i++)
         {
             for(int j=0; j<K;j++){
                 int n = h_queryidx[i*K+j];
               fprintf(csv_file_g,"%f,%f,%f,%d\n",h_output[(0*Samples+i)*K+j],h_output[(1*Samples+i)*K+j],h_output[(2*Samples+i)*K+j], 255);
             }
         }

//     for(int i=0; i<Samples; i++) //
//     {
//         for(int j=0; j<K;j++)
//         {
//             printf("[%f,%f,%f]",h_output[(0*Samples+i)*32+j],h_output[(1*Samples+i)*32+j],h_output[(2*Samples+i)*32+j]);
//         }
//         printf("\n");
//     }

}
