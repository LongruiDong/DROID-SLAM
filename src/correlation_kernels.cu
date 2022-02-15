#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// 使用宏AT_DISPATCH_FLOATING_TYPES_AND_HALF
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>

#define BLOCK 16

__forceinline__ __device__ bool within_bounds(int h, int w, int H, int W) {
  return h >= 0 && h < H && w >= 0 && w < W;
}
//参考cuda编程基础 核函数(在gpu上执行) 具体实现
template <typename scalar_t> //一个grid有12个Blocks 256线程
__global__ void corr_index_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> volume,//(1#,48,64,48/2^l,64/2^l)
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> coords,//(1#,2,48,64)
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> corr,//(1#,7,7,48,64)
    int r)//3
{
  // batch index 计算索引 blockDim.x= 16 blockDim.y= 16 正好和map尺寸对上 一个线程处理一个像素
  const int x = blockIdx.x * blockDim.x + threadIdx.x; //
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int n = blockIdx.z; // grid维度(4,3,#edge) 以上索引确定了 某条边 两帧之间每个像素！

  const int h1 = volume.size(1);//48
  const int w1 = volume.size(2);//64
  const int h2 = volume.size(3);//48/2^l
  const int w2 = volume.size(4);//64/2^l

  if (!within_bounds(y, x, h1, w1)) {//要保证 y<= h1 x<=w1 按照上面绝对索引计算理应在此范围
    return;
  }

  float x0 = coords[n][0][y][x];// y/2^l x/2^l                                                                                                                                                                                                                                                                                                                        能cover coords所有位置
  float y0 = coords[n][1][y][x];

  float dx = x0 - floor(x0);//小数部分 {0，0.5，0.75，0.125}
  float dy = y0 - floor(y0);

  int rd = 2*r + 1;//7
  for (int i=0; i<rd+1; i++) {//遍历8次 邻域网格
    for (int j=0; j<rd+1; j++) {
      int x1 = static_cast<int>(floor(x0)) - r + i;
      int y1 = static_cast<int>(floor(y0)) - r + j;

      if (within_bounds(y1, x1, h2, w2)) {//这里h2 w2是每个level大小
        scalar_t s = volume[n][y][x][y1][x1];//(y,x)上的相似性 在当前level下 在该邻域内某个像素的相似性

        if (i > 0 && j > 0)
          corr[n][i-1][j-1][y][x] += s * scalar_t(dx * dy);

        if (i > 0 && j < rd)
          corr[n][i-1][j][y][x] += s * scalar_t(dx * (1.0f-dy));

        if (i < rd && j > 0)
          corr[n][i][j-1][y][x] += s * scalar_t((1.0f-dx) * dy);

        if (i < rd && j < rd)//当只看level0时 就是C0 的值 当前像素各level相似性加权求和！
          corr[n][i][j][y][x] += s * scalar_t((1.0f-dx) * (1.0f-dy));

      }
    }
  }
}


template <typename scalar_t>
__global__ void corr_index_backward_kernel(
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> coords,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> corr_grad,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> volume_grad,
    int r)
{
  // batch index
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int n = blockIdx.z;

  const int h1 = volume_grad.size(1);
  const int w1 = volume_grad.size(2);
  const int h2 = volume_grad.size(3);
  const int w2 = volume_grad.size(4);

  if (!within_bounds(y, x, h1, w1)) {
    return;
  }

  float x0 = coords[n][0][y][x];
  float y0 = coords[n][1][y][x];

  float dx = x0 - floor(x0);
  float dy = y0 - floor(y0);

  int rd = 2*r + 1;
  for (int i=0; i<rd+1; i++) {
    for (int j=0; j<rd+1; j++) {
      int x1 = static_cast<int>(floor(x0)) - r + i;
      int y1 = static_cast<int>(floor(y0)) - r + j;

      if (within_bounds(y1, x1, h2, w2)) {
        scalar_t g = 0.0;
        if (i > 0 && j > 0)
          g += corr_grad[n][i-1][j-1][y][x] * scalar_t(dx * dy);

        if (i > 0 && j < rd)
          g += corr_grad[n][i-1][j][y][x] * scalar_t(dx * (1.0f-dy));

        if (i < rd && j > 0)
          g += corr_grad[n][i][j-1][y][x] * scalar_t((1.0f-dx) * dy);

        if (i < rd && j < rd)
          g += corr_grad[n][i][j][y][x] * scalar_t((1.0f-dx) * (1.0f-dy));

        volume_grad[n][y][x][y1][x1] += g;
      }
    }
  }
}
// corr.py droid_backends.corr_index_forward具体实现部分
std::vector<torch::Tensor> corr_index_cuda_forward(
    torch::Tensor volume, //(#edge,48,64,48/2^i,64/2^i) correlation volume 某一层
    torch::Tensor coords, //(#edge,2,48,64)
    int radius) //3
{
  const auto batch_size = volume.size(0);//1 #在factor graph上时 是边的数量 对于>1的情况 是怎么处理
  const auto ht = volume.size(1);//48
  const auto wd = volume.size(2);//64
  //BLOCK=16 下面和cuda有关  dim3 is a special CUDA datatype with 3 components .x, .y, .z each initialized to 1
  const dim3 blocks((wd + BLOCK - 1) / BLOCK, // 79/16 4.9-> 4
                    (ht + BLOCK - 1) / BLOCK, // 63/16 3.9-> 3 
                    batch_size); // 1 一个核函数只用一个grid no  batchsize直接体现 factorgraph 边的数量！
  //12个blocks 总尺寸上限与硬件的计算能力相关
  const dim3 threads(BLOCK, BLOCK); //(16,16) 每个block256个并行线程= 8个thread warp

  auto opts = volume.options(); //?
  torch::Tensor corr = torch::zeros( //(1,7,7,48,64)
    {batch_size, 2*radius+1, 2*radius+1, ht, wd}, opts);
  //通过pytorch提供的接口调用cuda kernel(本代码调用sampler_forward_kernel)
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(volume.type(), "sampler_forward_kernel", ([&] {
    corr_index_forward_kernel<scalar_t><<<blocks, threads>>>(//网格 grid 维度，线程块 block 维度:grid中block大小 和 每个block中thread的大小
      volume.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),// 5指维度
      coords.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
      corr.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
      radius);
   }));

  return {corr};

}

std::vector<torch::Tensor> corr_index_cuda_backward(
  torch::Tensor volume,
  torch::Tensor coords,
  torch::Tensor corr_grad,
  int radius)
{
  const auto batch_size = volume.size(0);
  const auto ht = volume.size(1);
  const auto wd = volume.size(2);

  auto volume_grad = torch::zeros_like(volume);

  const dim3 blocks((wd + BLOCK - 1) / BLOCK, 
                    (ht + BLOCK - 1) / BLOCK, 
                    batch_size);

  const dim3 threads(BLOCK, BLOCK);


  AT_DISPATCH_FLOATING_TYPES_AND_HALF(volume.type(), "sampler_backward_kernel", ([&] {
    corr_index_backward_kernel<scalar_t><<<blocks, threads>>>(
      coords.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
      corr_grad.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
      volume_grad.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
      radius);
   }));

  return {volume_grad};
}